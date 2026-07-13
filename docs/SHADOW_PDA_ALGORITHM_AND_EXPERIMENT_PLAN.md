# ShadowPDA / Track-Before-Recover：算法设计、反思与实验方案

> 项目：FM-Track / MOT20 online multi-object tracking  
> 当前目标：把低分恢复从“开关调参”升级为一个有新意、可解释、可实验验证的 online MOT 方法。  
> 建议方法名：**ShadowPDA: Probabilistic Shadow-State Recovery for Online MOT**  
> 中文名：**基于概率影子状态的在线身份恢复**  

---

## 0. 当前结论先写在前面

我们不应继续把方法做成：

```text
if release_ratio > threshold:
    enable OSR
else:
    disable OSR
```

这最多是一个工程保险开关，不足以作为论文主 idea。

真正要做的是：

```text
低分检测不是恢复动作，
而是隐藏身份在杂波环境下的概率证据。
```

因此，最终方法应从 **sequence-level switch** 转向 **proposal-level / identity-level shadow-state recovery**。

一句话定义：

```text
ShadowPDA 将低置信检测视为杂波环境下 lost identity 的概率证据，
通过在线影子状态累计、PDA 式软关联、clutter 假设和风险约束，
在证据足够可靠后才把隐藏身份恢复为公开轨迹。
```

---

## 1. 为什么不能只做开关

### 1.1 诊断混合不是主方法

我们曾经做过诊断混合：

```text
01 baseline
02 OSR / shadow-warmup
03 baseline
05 baseline
```

这种实验可以证明“OSR 在某些场景有潜力”，但不能作为论文结果，因为它依赖已知序列表现，容易被认为是：

```text
sequence-specific tuning
cherry-picking
oracle switch
```

### 1.2 Shadow warm-up 虽然自动，但仍然不够

Shadow warm-up 的思路是：先运行 shadow tracker，统计低分恢复后是否能等到 high-score release，再决定 main tracker 是否启用 OSR。

它的优点：

```text
02 能自动开启并涨分；
05 能自动拒绝并止损。
```

它的问题：

```text
03 明明 pending-high 有正收益，
但 release ratio 低，导致 shadow warm-up 误杀。
```

这说明：

```text
release ratio 是一个有用风险信号，
但不是完整的恢复可靠性模型。
```

因此，shadow warm-up 应保留为诊断证据或辅助保险，但不能作为最终核心算法。

---

## 2. 文献启发：手工规则可以，但必须有建模重定义

近几年很多 online MOT 方法表面上是规则，但真正贡献不是“多加阈值”，而是重新定义了某个跟踪问题。

### 2.1 ByteTrack / ByteTrackV2

表面规则：

```text
高分框先关联，低分框再关联。
```

真正 idea：

```text
低分检测中仍包含真实目标，不能直接丢弃；
要通过层级关联挖掘低分框中的 true positive。
```

对我们的启发：

```text
低分框有价值，但不能粗暴丢弃。
```

### 2.2 OC-SORT / Deep OC-SORT

表面规则：

```text
observation-centric recovery / motion correction。
```

真正 idea：

```text
遮挡期间不能一直相信 Kalman 预测；
遮挡后的恢复需要回到 observation 修正 accumulated error。
```

对我们的启发：

```text
lost 后的恢复不能只靠主状态预测；
低分观测可以作为修正线索，但要小心其噪声。
```

### 2.3 Hybrid-SORT

表面规则：

```text
加入 confidence、height、velocity direction 等 weak cues。
```

真正 idea：

```text
遮挡和聚集时，spatial / appearance 这些强线索会同时变模糊；
weak cues 可以补偿强线索失效。
```

对我们的启发：

```text
低分恢复不能只看 ReID 或 IoU，
需要把 weak evidence 和 risk 显式建模。
```

### 2.4 UCMCTrack

表面规则：

```text
uniform camera motion compensation，ground-plane Kalman filter，Mapped Mahalanobis Distance。
```

真正 idea：

```text
换一个几何/运动空间，运动不确定性会更合理。
```

对我们的启发：

```text
我们也应换恢复决策空间：
从 public association 转到 hidden identity evidence estimation。
```

### 2.5 C-BIoU

表面规则：

```text
small buffer / large buffer cascaded IoU matching。
```

真正 idea：

```text
当运动不规则、外观相似时，需要扩展匹配空间，
但必须分级控制过度扩张风险。
```

对我们的启发：

```text
低分恢复也是在扩展 recovery space；
必须有风险控制，否则 FP / IDSW 会爆。
```

### 2.6 BoostTrack++

表面规则：

```text
用 tracklet context、shape、Mahalanobis、soft BIoU 重新 boost detection confidence。
```

真正 idea：

```text
detector score 不是固定真理，
可以被 tracklet context 重新校准。
```

对我们的启发：

```text
低分检测不应只看 detector score；
应该看它是否是某个 lost identity 的概率证据。
```

### 2.7 雷达 PDA / JPDA / Track-Before-Detect

雷达多目标跟踪面临：

```text
false alarm / clutter
missed detection
data association uncertainty
track initiation / confirmation / deletion
```

PDA/JPDA 的核心不是硬选一个测量，而是在候选测量和 false alarm / missed detection 假设之间计算关联概率。

Track-before-detect 的核心是：

```text
单帧信号太弱时，不先宣布目标成立，
而是先在时间上累计弱证据，
等证据足够后再确认目标。
```

对应到 MOT：

```text
低分检测 = 弱量测 / 杂波 / clutter
lost track = 暂时失联的航迹
ID recovery = track confirmation
误恢复 = 把 clutter 当成目标
```

因此，我们的核心应是：

```text
Track-Before-Recover：先跟踪低分证据，再恢复身份。
```

---

## 3. 方法总目标

我们要解决的问题不是：

```text
低分框要不要参与恢复？
```

而是：

```text
低分框在 clutter 和遮挡环境下，
怎样作为 lost identity 的概率证据被确认？
```

正式问题定义：

```text
Given an online tracker with Tracked/Lost/Removed states,
we model low-confidence recovery as a clutter-aware hidden identity confirmation problem.
```

中文：

```text
给定一个在线跟踪器，我们将低置信恢复建模为杂波感知的隐藏身份确认问题。
```

---

## 4. ShadowPDA 的核心创新点

### 4.1 观测角色重定义

传统方法：

```text
low-score detection → association observation → possible recovery
```

ShadowPDA：

```text
low-score detection → probabilistic evidence → shadow state update → possible recovery
```

核心句：

```text
Low-confidence detections are evidence, not immediate public observations.
```

### 4.2 状态空间重定义

传统 tracker：

```text
Tracked → Lost → Removed
```

ShadowPDA：

```text
Tracked
Lost
ShadowTentative
ShadowConfirmed
PublicRecovered
Removed
```

其中：

```text
ShadowTentative / ShadowConfirmed 不输出、不污染主 Kalman、不直接改变 public trajectory。
```

### 4.3 关联方式重定义

传统 OSR：

```text
best low detection hard match
```

ShadowPDA：

```text
PDA-style soft association with clutter hypothesis
```

即：

```text
多个低分候选分别有多大概率属于 lost identity？
还有多大概率当前其实是 missed detection / clutter？
```

### 4.4 恢复动作重定义

传统：

```text
低分匹配成功 → re_activate
```

ShadowPDA：

```text
低分证据累计 → shadow reliability 上升；
只有高分确认或极低风险条件满足，才 public recover。
```

---

## 5. ShadowState 数据结构

建议新增：

```python
@dataclass
class ShadowState:
    track_id: int
    start_frame: int
    last_frame: int
    age: int
    support_count: int
    miss_count: int

    existence_logit: float
    reliability: float

    ghost_tlbr: np.ndarray
    ghost_feat: np.ndarray | None
    last_low_det_idx: int

    p_best: float
    p_second: float
    p_clutter: float
    entropy: float
    margin: float
    risk: float

    best_evidence: float
    avg_evidence: float
    avg_memory: float
    avg_det_score: float

    state: str  # tentative / confirmed
```

最重要变量：

```text
existence_logit: 隐藏身份存在性证据
reliability = sigmoid(existence_logit)
```

---

## 6. 每帧算法流程

### 6.1 原始检测拆分

```text
high_dets = detections with score >= track_high_thresh
low_dets  = detections with track_low_thresh < score < track_high_thresh
```

### 6.2 正常主关联

先执行 baseline 主流程：

```text
Tracked/Lost 与 high_dets 做 primary association
Tracked 与 low_dets 做原有 second matching
```

注意：

```text
low_dets 不直接恢复 lost track。
```

### 6.3 对 unmatched lost tracks 更新 ShadowState

对每个 unmatched lost track `i`：

1. 在 low_dets 中找候选。
2. 对每个候选 `d_j` 计算 evidence score。
3. 加入 clutter / missed detection 假设。
4. 做 PDA-style soft association。
5. 更新 existence_logit、ghost_box、risk。
6. 根据 reliability 做 shadow 状态转移。

---

## 7. Evidence Score 设计

对 lost track `i` 和低分检测 `d_j`：

```text
e_ij =
    w_app * app_sim(i, j)
  + w_mot * motion_score(i, j)
  + w_geo * geometry_score(i, j)
  + w_det * det_score(j)
  + w_q   * track_quality(i)
  - w_age * lost_age_norm(i)
  - w_occ * active_overlap_risk(i, j)
```

### 7.1 app_sim

```text
app_sim = max(sim(smooth_feat_i, feat_j), top-k feature bank similarity)
```

原因：

```text
smooth_feat 可能被遮挡前状态污染；
feature bank top-k 更鲁棒。
```

### 7.2 motion_score

```text
center_step = distance(center(det), center(pred_or_ghost)) / max(1, lost_age)
motion_score = exp(-center_step / max_center_step)
```

### 7.3 geometry_score

包含：

```text
area ratio
height ratio
aspect ratio consistency
```

### 7.4 det_score

将低分检测置信度归一化：

```text
det_score_norm = (score - track_low_thresh) / (track_high_thresh - track_low_thresh)
```

### 7.5 track_quality

```text
track_quality = f(track length, recent score, feature bank size, recent stability)
```

### 7.6 active_overlap_risk

如果低分框与当前 active tracked person 高重叠，则风险增加，但不要硬拒绝：

```text
active_overlap_risk = max IoU / IoA with active tracks
```

原因：之前实验证明 active overlap 硬 gate 会杀掉 02 的有效恢复。

---

## 8. PDA-style Soft Association

对 lost track `i` 的所有低分候选，计算：

```text
p_ij = exp(e_ij / τ) / (exp(e_i0 / τ) + Σ_k exp(e_ik / τ))
```

其中：

```text
e_i0 = clutter / missed detection hypothesis
```

`p_i0`：

```text
p_i0 = exp(e_i0 / τ) / (exp(e_i0 / τ) + Σ_k exp(e_ik / τ))
```

解释：

```text
p_ij: 低分检测 j 属于 lost identity i 的概率
p_i0: 当前没有可靠低分观测，或候选都是 clutter 的概率
```

这一步是 ShadowPDA 的核心，不是普通 hard matching。

---

## 9. 关联不确定性与风险

### 9.1 Entropy

```text
H_i = -Σ_j p_ij log(p_ij + eps)
```

高 entropy 表示多个候选都像该 track，当前关联歧义高。

### 9.2 Margin

```text
margin_i = p_best - p_second
```

低 margin 表示 best 和 second 候选差距小，恢复风险高。

### 9.3 Risk

```text
risk_i =
    λ_H * H_i
  + λ_m * max(0, margin_min - margin_i)
  + λ_o * active_overlap_risk
  + λ_a * lost_age_norm
```

这比 `max_ambiguity <= 1` 更有解释性。

---

## 10. Existence Logit 更新

核心更新：

```text
L_i^t =
    ρ * L_i^{t-1}
  + Σ_j p_ij * e_ij
  - λ_0 * p_i0
  - risk_i
```

其中：

```text
L_i^t: existence_logit
ρ: temporal decay
p_i0: clutter / missed detection probability
```

如果无候选：

```text
L_i^t = ρ * L_i^{t-1} - miss_penalty
```

可靠性：

```text
R_i^t = sigmoid(L_i^t)
```

---

## 11. Ghost Box 更新

不更新主 Kalman，只更新 shadow box：

```text
ghost_box_i^t =
    (1 - η) * ghost_box_i^{t-1}
  + η * Σ_j p_ij * box_j
```

其中：

```text
η = min(0.5, p_best)
```

如果当前候选很不确定，ghost_box 只小幅移动。

---

## 12. 状态转移

### 12.1 Lost → ShadowTentative

```text
if reliability > θ_tentative and support_count >= 1:
    state = ShadowTentative
```

### 12.2 ShadowTentative → ShadowConfirmed

```text
if reliability > θ_confirm
and support_count >= M
and entropy < θ_entropy
and margin > θ_margin:
    state = ShadowConfirmed
```

### 12.3 ShadowConfirmed → PublicRecovered

两种方式：

#### A. 高分检测确认

当后续 high-score detection `h` 与 shadow state 匹配：

```text
score_high(i, h) =
    α * app_sim(i, h)
  + β * motion_to_ghost(i, h)
  + γ * det_score(h)
  - δ * risk_i
```

若：

```text
score_high > θ_high
```

则：

```text
track.re_activate(high_det)
```

#### B. 极低风险低分恢复

只在非常严格条件下允许：

```text
reliability > θ_public
p_best > 0.85
entropy < 0.15
margin > 0.45
support_count >= 3
app_sim_best > 0.85
det_score_best > 0.35
lost_age <= 7
active_overlap_risk < 0.3
```

则：

```text
track.re_activate(best_low_det)
```

这不是 ByteTrack 式低分恢复，而是：

```text
PDA-confirmed low recovery
```

---

## 13. 完整伪代码

```text
for each frame t:
    split detections into high_dets and low_dets

    # Stage 1: normal high-score association
    match tracked/lost tracks with high_dets

    # Stage 2: normal low-score update for still tracked tracks only
    match unmatched tracked tracks with low_dets

    # Stage 3: ShadowPDA for unmatched lost tracks
    for each unmatched lost track i:
        candidates = gated low_dets

        if candidates empty:
            L_i = decay * L_i - miss_penalty
            update miss_count
            continue

        for each candidate d_j:
            e_ij = evidence(i, d_j)

        compute clutter hypothesis e_i0
        compute p_ij and p_i0 using softmax
        compute entropy, margin, risk

        L_i = decay * L_i + Σ p_ij * e_ij - clutter_penalty * p_i0 - risk
        R_i = sigmoid(L_i)
        update ghost_box using weighted candidate boxes
        update support_count / miss_count

        if R_i > theta_confirm and risk low:
            mark ShadowConfirmed

    # Stage 4: high-score recovery through shadow state
    for each ShadowConfirmed state:
        try matching remaining high_dets to ghost_box
        if matched:
            re_activate(track, high_det)
            delete shadow

    # Stage 5: ultra-safe low-score public recovery
    for each ShadowConfirmed state:
        if R_i > theta_public and risk very low:
            re_activate(track, best_low_det)
            delete shadow

    # Stage 6: delete stale shadows
    if L_i < theta_delete or miss_count > max_miss or age > max_age:
        delete shadow
```

---

## 14. 第一版参数建议

```text
# PDA evidence
rho = 0.85
tau = 0.12
clutter_penalty = 0.8
miss_penalty = 0.4
entropy_penalty = 0.6
margin_min = 0.25

# Evidence weights
w_app = 1.2
w_mot = 0.8
w_geo = 0.4
w_det = 0.4
w_q = 0.3
w_age = 0.3
w_occ = 0.5

# State thresholds
theta_tentative = 0.60  # reliability
theta_confirm = 0.78
theta_public = 0.93
theta_delete_logit = -1.5

# Support
min_support_confirm = 2
min_support_public = 3
max_shadow_age = 15
max_shadow_miss = 3

# High confirmation
high_min_score = 0.60
high_min_app = 0.70
high_max_center_step = 120
high_min_fusion_score = 0.60

# Ultra-safe low public recovery
ultra_min_det_score = 0.35
ultra_min_app = 0.85
ultra_max_entropy = 0.15
ultra_min_margin = 0.45
ultra_max_lost_age = 7
ultra_max_active_overlap = 0.30
```

这些不是最终调参，而是第一版 sanity 起点。

---

## 15. 与已有版本的关系

### v1 Instant OSR

```text
low det → immediate recovery
```

用途：负面消融。

### v2 Pending-high

```text
low det re_activate，但暂时不输出，等 high release。
```

用途：说明低分桥接有价值，但会污染主状态。

### v3 Ghost Proposal

```text
low det 只做 ghost，等 high recovery。
```

用途：说明完全不污染主状态很安全，但太保守。

### v4 Shadow Warm-up

```text
sequence-level shadow reliability gate。
```

用途：说明 release ratio 有诊断价值，但不是最终算法。

### v5 ShadowPDA

```text
proposal-level probabilistic shadow-state recovery。
```

目标主方法。

---

## 16. 实现计划

新增文件：

```text
scripts/dmm_base_tracker_shadow_pda.py
```

不要修改：

```text
scripts/dmm_base_tracker.py
scripts/dmm_base_tracker_osr.py
scripts/dmm_base_tracker_osr_confirm.py
scripts/dmm_base_tracker_osr_ghost.py
scripts/dmm_base_tracker_osr_shadow_warmup.py
```

### 16.1 第一阶段：最小可运行版本

实现：

```text
ShadowState dataclass
low candidate evidence scoring
softmax with clutter hypothesis
existence_logit update
entropy / margin risk
ghost_box update
high-score recovery
summary stats
```

先不启用 ultra-safe low recovery。

目标：

```text
确认 ShadowPDA 能创建 / 更新 / 确认 shadow state；
确认不会污染主 tracker；
确认 05 不爆 FP/IDSW。
```

### 16.2 第二阶段：high-score shadow recovery

加入：

```text
ShadowConfirmed + high_det → public recovery
```

目标：

```text
看 02 是否能接近或超过 shadow-warmup / pending-high。
```

### 16.3 第三阶段：ultra-safe low recovery

加入极低风险低分恢复。

目标：

```text
解决 ghost 太保守、high recovery 太少的问题；
重点观察 03 是否能恢复正收益。
```

### 16.4 第四阶段：消融与稳定性

只在前面 smoke 成功后再做全量。

---

## 17. 实验方案总览

### 17.1 数据集

主验证：

```text
MOT20 train: MOT20-01, MOT20-02, MOT20-03, MOT20-05
```

后续可扩展：

```text
MOT17 train
DanceTrack val
```

当前先不要扩展，先把 MOT20 train 证明清楚。

---

## 18. 实验一：Smoke Test

### 18.1 MOT20-02 前 1000 帧

命令目标：

```text
确认 shadow state 正常工作。
```

记录指标：

```text
shadow_created
shadow_updated
shadow_confirmed
shadow_deleted
high_recovered
low_public_recovered
mean_entropy
mean_margin
mean_reliability
rows
unique_tracks
```

通过条件：

```text
shadow_created > 0
shadow_confirmed > 0
程序不崩溃
rows 与 baseline 接近
```

### 18.2 MOT20-05 前 1000 帧

目标：

```text
确认 05 不会大量错误恢复。
```

通过条件：

```text
FP 不明显增加
IDSW 不明显增加
low_public_recovered 接近 0
```

---

## 19. 实验二：单序列 full

### 19.1 MOT20-02 full

对比：

```text
baseline HOTA = 68.821
pending-high HOTA = 68.860
shadow-warmup HOTA = 68.956
```

目标：

```text
ShadowPDA HOTA >= 68.86
最好接近或超过 68.956
IDF1 >= baseline
FP 不明显增加
```

### 19.2 MOT20-03 full

对比：

```text
baseline HOTA = 80.013
pending-high HOTA = 80.053
```

目标：

```text
ShadowPDA 不应像 release-ratio warm-up 那样误杀 03；
HOTA 应 >= baseline，最好接近 pending-high。
```

### 19.3 MOT20-05 full

对比：

```text
baseline HOTA = 78.735
pending-high HOTA = 78.623
adaptive HOTA = 78.539
```

目标：

```text
ShadowPDA HOTA >= 78.70
FP / IDSW / Frag 不明显超过 baseline
```

### 19.4 MOT20-01 full

对比：

```text
baseline HOTA = 77.148
pending-high / confirm 类方法会下降
```

目标：

```text
ShadowPDA 不明显伤害 01；
如果无有效恢复，自动保持接近 baseline。
```

---

## 20. 实验三：全 train

对比方法：

```text
1. baseline
2. OSR v1 instant
3. OSR v2 confirm / pending-high
4. OSR v3 ghost-only
5. OSR v4 shadow-warmup
6. ShadowPDA high-only
7. ShadowPDA full
```

主指标：

```text
HOTA
IDF1
AssA
DetA
MOTA
IDSW
Frag
FP
FN
```

成功条件：

```text
HOTA > baseline 77.812
IDF1 > baseline 89.438
FP 不显著增加
IDSW 不显著增加
```

最低可接受结果：

```text
HOTA 小幅正收益，IDF1 正收益，FP/IDSW 不爆。
```

理想结果：

```text
同时保留 02/03 收益，避免 01/05 伤害。
```

---

## 21. 消融实验设计

### 21.1 去掉 clutter hypothesis

```text
ShadowPDA w/o p_i0
```

目的：证明 clutter / missed detection 假设重要。

预期：

```text
05 FP/IDSW 更高。
```

### 21.2 去掉 entropy risk

```text
ShadowPDA w/o entropy
```

目的：证明关联歧义建模重要。

预期：

```text
拥挤场景 IDSW 增加。
```

### 21.3 hard best candidate vs PDA soft association

```text
best-low hard update
vs
soft PDA update
```

目的：证明不是普通 hard matching。

预期：

```text
soft association 更稳，尤其在 05。
```

### 21.4 high-only recovery vs full recovery

```text
ShadowPDA high-confirm only
ShadowPDA high-confirm + ultra-safe low recovery
```

目的：证明 ultra-safe low recovery 是否必要。

预期：

```text
high-only 更安全但收益有限；
full 可能提升 02/03，但需要控制 05。
```

### 21.5 去掉 active overlap risk

目的：证明 active overlap 不能硬 gate，但作为 soft risk 有价值。

预期：

```text
去掉后 05 错误恢复增加。
```

---

## 22. 事件审计方案

为了避免只看总指标，需要保存事件 CSV：

```text
frame
track_id
shadow_state
existence_logit
reliability
support_count
miss_count
p_best
p_second
p_clutter
entropy
margin
risk
best_det_score
best_app_sim
best_motion_score
active_overlap
public_recovered
recovery_type  # high / low_ultra
```

GT audit：

```text
recovered event 是否匹配正确 GT
low candidate 是否本身是 FP
track historical majority GT 是否一致
```

对比 02 / 05：

```text
02：高 reliability 事件应多数正确
05：高 risk / p_clutter 应能过滤错误事件
```

---

## 23. 当前已有实验事实，应写入论文动机

### 23.1 Instant recovery 失败

```text
FN 降，但 FP / IDSW / Frag 增，HOTA 降。
```

结论：

```text
低分框有召回价值，但不能直接公开恢复。
```

### 23.2 Pending-high 局部有效

```text
02 / 03 正收益；01 / 05 负收益。
```

结论：

```text
低分框作为桥接证据有价值，但状态污染风险存在。
```

### 23.3 Ghost-only 太保守

```text
不污染主状态，但 high recovery 太少，收益不足。
```

结论：

```text
完全等待高分确认过于保守，需要概率证据确认。
```

### 23.4 Shadow warm-up 有诊断价值但不足

```text
02 能识别，05 能拒绝，03 被误杀。
```

结论：

```text
sequence-level reliability 不够，必须转向 proposal-level evidence。
```

---

## 24. 论文主线建议

标题方向：

```text
ShadowPDA: Probabilistic Shadow-State Recovery for Online Multi-Object Tracking
```

或：

```text
Track-Before-Recover: Clutter-Aware Shadow Confirmation for Online MOT
```

摘要核心句：

```text
Low-confidence detections are often discarded or directly associated in existing trackers. However, in crowded scenes they are neither pure noise nor immediately reliable observations. We propose to treat them as probabilistic evidence for hidden identities under clutter.
```

贡献点：

```text
1. We reformulate low-score recovery as clutter-aware hidden identity confirmation.
2. We introduce shadow states that accumulate low-confidence evidence without contaminating public trajectories.
3. We design a PDA-style soft association with clutter hypothesis and ambiguity risk for shadow reliability estimation.
4. We recover identities only through high-score confirmation or ultra-safe low-score confirmation, preserving online causality.
```

---

## 25. 风险与反思

### 25.1 风险：公式看起来复杂但还是规则

解决：

```text
必须用 ablation 证明每个组件有作用：p_i0、entropy、soft association、shadow state。
```

### 25.2 风险：收益很小

解决：

```text
先追求稳定超过 baseline；
再优化 02/03 的 recall。
```

### 25.3 风险：05 仍被伤害

解决：

```text
先 high-confirm only，确认安全；
再开启 ultra-safe low recovery。
```

### 25.4 风险：03 仍没有收益

解决：

```text
分析 03 pending-high 正收益事件；
看它们是否来自 low public recovery 而非 high release；
相应放宽 ultra-safe low recovery 的 support / risk 条件。
```

---

## 26. 下一步执行清单

### Step 1：新增 ShadowPDA 脚本

```text
scripts/dmm_base_tracker_shadow_pda.py
```

### Step 2：实现 high-confirm only 版本

先不允许 low public recovery。

### Step 3：MOT20-02 1000 帧 smoke

确认 shadow stats 正常。

### Step 4：MOT20-05 1000 帧 smoke

确认不爆。

### Step 5：MOT20-02 full + eval

目标：超过 baseline，接近 pending-high。

### Step 6：MOT20-03 full + eval

目标：不要误杀 03。

### Step 7：开启 ultra-safe low recovery

观察 03 是否提升。

### Step 8：MOT20 train all

统一参数，全自动，不看序列名。

---

## 27. 最终判断

当前我们应该停止把精力放在 sequence-level switch 上。

最终 idea 应该是：

```text
ShadowPDA / Track-Before-Recover
```

核心不是调开关，而是：

```text
将低分恢复从 deterministic association 问题，
重新建模为 clutter-aware probabilistic shadow-track confirmation 问题。
```

这才有新意，也能解释我们已有的全部实验现象。
