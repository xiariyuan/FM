# TrustTrack / ShadowPDA 会话交接文档 — 2026-07-10

> 新会话请先阅读本文件，再继续执行。  
> 工作目录：`/gemini/code/FMtrack-main/FM-Track`  
> 当前项目：MOT20 / FM-Track / ShadowPDA / TrustTrack 方向实验。

---

## 0. 最重要结论

本轮会话的核心结论已经发生更新：

```text
不要继续把 ShadowPDA-V3 的 low public recovery 当主线。
ShadowPDA 可以保留为 lost-track fallback。
真正要提升 HOTA / IDF1 / AssA，主线应转向 TrustTrack / cue ambiguity-aware association。
```

但 TrustTrack 的原始命题也必须修正：

```text
原命题：线索可靠性反转 Reliability Reversal 导致关联错误。
新结论：局部线索区分力塌陷 cue collapse / ambiguity 更强、更稳定。
```

当前最可靠的统计信号是：

```text
cue_collapse = 1 - max_pair_margin
app_col_margin
motion_col_margin
max_pair_margin
```

而不是原始 `RRS`。

---

## 1. 用户目标

用户明确要求：

```text
不仅仅是 63.6 → 64/65，能冲更高更好；
如果内部验证能到 HOTA 80+ 最好；
同时需要深入思考是否可以基于当前权重提升目标检测效果。
```

需要注意：

```text
如果指官方 MOT20 test leaderboard，HOTA 80+ 不现实；当前榜首也只有约 67。
如果指内部 train split 或自建验证集，HOTA 80+ 也不能靠 low-score recovery，需要检测 + 关联系统级改造。
```

短期主线：

```text
AssA / IDF1 / IDSW
```

长期主线：

```text
association + detector postprocess + tracklet-conditioned detection boost + learned association / RelateTrack
```

---

## 2. 之前 ShadowPDA 实验结论

### 2.1 ShadowPDA-V3 / V3-Precision 总结

ShadowPDA-V3 主要处理：

```text
lost track 的低分检测是否允许 public recovery
```

现有结论：

```text
MOT20-02：V3 很干净但收益小。
MOT20-03：low public recovery 的 det_fp 与 correct 难区分。
MOT20-05：V3 有一定正收益，但 V3-Precision 收益小。
MOT20-01：基本不触发，指标中性。
```

关键教训：

```text
low public recovery 是局部补救，不足以大幅提升 HOTA。
继续调 V3 output_score / state_margin / release threshold 意义不大。
```

### 2.2 high-only 消融

之前已经做过 high-only vs no-recovery 文件级对比：

```text
MOT20-01 identical
MOT20-03 identical
MOT20-05 identical
```

结论：

```text
primary_high_recovered 只是内部统计，不改变输出，不能作为主贡献。
```

---

## 3. MOT20 leaderboard 对路线的启发

用户贴过 MOT20 test 排行榜。

Rank 29 旧 idea：

```text
HOTA = 63.6
IDF1 = 77.8
AssA = 63.2
DetA = 64.1
IDSW = 1359
```

Rank 2 Public Detections：

```text
HOTA = 66.5
IDF1 = 82.7
AssA = 69.2
DetA = 64.0
IDSW = 660
```

最关键对比：

```text
Rank29 DetA = 64.1
Rank2  DetA = 64.0

Rank29 AssA = 63.2
Rank2  AssA = 69.2
```

所以：

```text
63.6 → 64/65/66 的主要瓶颈是 association / ID consistency，不是 DetA。
如果要冲更高，检测也会变成上限瓶颈。
```

---

## 4. TrustTrack 诊断实验

### 4.1 诊断脚本

已写入：

```text
scripts/trusttrack_rrs_diagnostic.py
scripts/trusttrack_rrs_diagnostic_lite.py
scripts/trusttrack_geo_collapse_lite.py
outputs/analyze_rrs_csv.py
```

用途：

```text
从已有 MOT result + detection dump + GT 离线重构每次匹配的 cue margin，
分析 RRS / cue collapse / pair margin 是否预测 wrong_id / det_fp / future switch。
```

重要：这些是离线重构，不是真实 tracker 内部 matrix；后续最好接入 tracker 内部真实 association logger。

---

## 5. MOT20-02 诊断结果

输出：

```text
outputs/trusttrack_rrs_m02_norecovery.csv
outputs/analyze_rrs_m02.out
```

总量：

```text
145060 rows
correct  = 138672
det_fp   = 1615
wrong_id = 4715
unknown  = 58
```

### 5.1 原始 RRS 不单调

```text
rrs = 0:
bad_rate ≈ 7.32%

rrs 0.193 ~ 0.358:
bad_rate ≈ 5.41%

rrs 0.358 ~ 0.434:
bad_rate ≈ 3.24%

rrs 0.434 ~ 0.555:
bad_rate ≈ 2.34%

rrs 0.555 ~ 0.893:
bad_rate ≈ 5.51%
```

结论：

```text
原始 RRS 不是稳定风险信号，不能直接作为主算法 gate。
```

### 5.2 cue_collapse 很强

```text
collapse 0.001 ~ 0.206:
bad_rate = 0.45%

collapse 0.524 ~ 0.675:
bad_rate = 7.28%

collapse 0.784 ~ 0.843:
bad_rate = 16.43%

collapse 0.843 ~ 0.951:
bad_rate = 26.44%

collapse 0.951 ~ 1.335:
bad_rate = 51.14%
```

结论：

```text
cue collapse / max_pair_margin 是强信号。
```

### 5.3 app_col_margin 很强

MOT20-02 `app_col_margin` 最低 10%：

```text
bad_rate = 26.94%
wrong/FP rate = 18.13%
```

结论：

```text
很多错误来自：同一个 detection 同时像多个 track。
因此 col_margin 必须作为核心。
```

---

## 6. MOT20-03 诊断结果

输出：

```text
outputs/trusttrack_rrs_m03_norecovery.csv
outputs/analyze_rrs_m03.out
```

总量：

```text
302553 rows
bad = 7733
wrongfp = 5619
```

### 6.1 原始 RRS 仍不单调

```text
rrs = 0:
bad_rate ≈ 2.60%

rrs 0.313 ~ 0.449:
bad_rate ≈ 1.17%

rrs 0.449 ~ 0.519:
bad_rate ≈ 0.87%

rrs 0.519 ~ 0.638:
bad_rate ≈ 1.15%

rrs 0.638 ~ 0.978:
bad_rate ≈ 2.61%
```

### 6.2 cue_collapse 仍很强

```text
collapse 0.001 ~ 0.116:
bad_rate = 0.48%

collapse 0.489 ~ 0.642:
bad_rate = 3.49%

collapse 0.642 ~ 0.732:
bad_rate = 9.22%

collapse 0.732 ~ 0.884:
bad_rate = 16.40%

collapse 0.884 ~ 1.343:
bad_rate = 34.70%
```

### 6.3 app/motion col margin 也强

MOT20-03 `app_col_margin` 最低 10%：

```text
bad_rate = 14.50%
wrong/FP rate = 10.86%
```

MOT20-03 `motion_col_margin` 最低 10%：

```text
bad_rate = 14.04%
wrong/FP rate = 10.45%
```

---

## 7. MOT20-05 诊断状态

尝试了 full 05 诊断：

```text
outputs/trusttrack_rrs_m05_norecovery.csv
outputs/trusttrack_rrs_m05_norecovery.out
```

结果：

```text
CSV 未生成，out 文件为空。
```

后续又写了轻量版：

```text
scripts/trusttrack_rrs_diagnostic_lite.py
scripts/trusttrack_geo_collapse_lite.py
```

但 MOT20-05 读取/筛选大 detection dump 仍然容易超时。

结论：

```text
05 诊断未完成。
不要在新会话里重复启动多个 full 05 诊断。
如果要做，先裁剪小 dump 或优化读取。
```

建议：

```text
先完成 02/03 的方法验证，再回头做 05。
```

---

## 8. TrustTrack-lite 最小实现

已写入：

```text
scripts/dmm_base_tracker_trusttrack_lite.py
```

### 8.1 设计目标

最小 intervention：

```text
不改 Hungarian matching
不改 Kalman update
不改检测框
不改生命周期
只在高 ambiguity 匹配时跳过 ReID feature update
```

核心思想：

```text
High-ambiguity matches may be acceptable for short-term box continuity,
but should not contaminate long-term identity memory.

高歧义匹配可以暂时维持短期轨迹，
但不应该污染长期 ReID 记忆。
```

### 8.2 实现方式

通过 monkeypatch `DMMTrack.update`：

```text
如果 (track_id, det_global_idx) 命中 freeze map：
    临时把 new_track.curr_feat 置 None
    调用原始 update()
    恢复 curr_feat
```

这样：

```text
Kalman / box / score / det_global_idx 照常更新；
只有 ReID feature 不更新。
```

### 8.3 gate 定义

每帧 tracker.update 前：

```text
active tracks × current detections
```

计算：

```text
appearance
motion
IoU
shape
```

四类 cue 的 row/col/pair margin。

定义：

```text
pair_margin_c = min(row_margin_c, col_margin_c)
trust = max_c pair_margin_c
collapse = 1 - trust
```

冻结条件：

```text
collapse >= trust-collapse-thresh
or app_col_margin <= trust-app-col-thresh
or motion_col_margin <= trust-motion-col-thresh
```

---

## 9. TrustTrack-lite smoke test

### 9.1 编译

已通过：

```text
python3 -m py_compile scripts/dmm_base_tracker_trusttrack_lite.py
```

### 9.2 MOT20-02 前 100 帧 smoke test

参数：

```text
--trust-lite-enable
--trust-min-det-score 0.5
--trust-collapse-thresh 0.80
--trust-app-col-thresh 0.08
--trust-motion-col-thresh 0.08
```

结果：

```text
rows = 3871
unique_tracks = 57
candidate_pairs = 149009
freeze_pairs_predicted = 145792
feature_updates_frozen = 540
feature_updates_normal = 3287
```

说明：

```text
功能可运行，但 freeze_pairs_predicted 很大，因为很多未匹配 pair 也被判风险；
真正生效的是 feature_updates_frozen。
```

---

## 10. TrustTrack-lite full MOT20-02 实验

已写/运行：

```text
outputs/run_trust_lite_m02_full_primary.sh
outputs/trust_lite_m02_full_primary/
```

参数：

```text
--trust-lite-enable
--trust-min-det-score 0.6
--trust-collapse-thresh 0.85
--trust-app-col-thresh 0.05
--trust-motion-col-thresh 0.05
```

这是 primary-only / 高分候选的保守版本。

### 10.1 Trust-lite full 02 结果

输出：

```text
outputs/trust_lite_m02_full_primary/MOT20-02_summary.json
outputs/trust_lite_m02_full_primary/eval.out
outputs/trust_lite_m02_full_primary/track_results/MOT20-02.txt
```

summary：

```text
rows = 145586
unique_tracks = 467
trust_stats:
  candidate_pairs = 8225821
  feature_updates_frozen = 41174
  feature_updates_normal = 104358
  frames = 2782
  freeze_pairs_predicted = 8125204
```

指标：

```text
HOTA = 68.047
DetA = 78.878
AssA = 58.816
IDF1 = 74.723
MOTA = 90.789
IDSW = 419
FN = 11521
FP = 2313
Frag = 759
```

### 10.2 同脚本 no-recovery baseline 对比

重新用同一个 eval 脚本评估 baseline：

```text
outputs/trust_lite_m02_baseline_eval/eval.out
```

baseline 指标：

```text
HOTA = 68.748
DetA = 78.844
AssA = 60.053
IDF1 = 75.205
MOTA = 90.750
IDSW = 428
FN = 11556
FP = 2329
Frag = 765
```

### 10.3 差异

Trust-lite primary-only 相对 baseline：

```text
HOTA: 68.047 - 68.748 = -0.701
AssA: 58.816 - 60.053 = -1.237
IDF1: 74.723 - 75.205 = -0.482
IDSW: 419 - 428 = -9    # IDSW 少了 9，是正向
FN: 11521 - 11556 = -35 # FN 少了 35，是正向
FP: 2313 - 2329 = -16   # FP 少了 16，是正向
Frag: 759 - 765 = -6    # Frag 少了 6，是正向
```

非常关键：

```text
虽然 IDSW/FN/FP/Frag 都略好，
但 AssA 和 IDF1 下降，导致 HOTA 明显下降。
```

说明当前 memory freeze 过强或错误冻结了有用 appearance update。

---

## 11. 对 TrustTrack-lite 的反思

### 11.1 memory freeze 不是当前版本的正向策略

当前 primary-only freeze：

```text
feature_updates_frozen = 41174
feature_updates_normal = 104358
```

冻结比例约：

```text
41174 / (41174 + 104358) ≈ 28.3%
```

这个比例太高。

结果表现：

```text
IDSW 下降一点，但 AssA / IDF1 大幅下降。
```

可能原因：

```text
1. ReID memory 在 MOT20-02 是有用的，冻结太多导致 identity representation 变旧。
2. freeze map 是对所有 candidate pairs 预判，再由 matched pair 命中，仍可能过宽。
3. cue-collapse 离线诊断能预测 bad match，但不代表 freeze memory 是最优动作。
4. 高 ambiguity 匹配不一定是错误；如果冻结真实更新，后续 appearance 会变差。
```

### 11.2 TrustTrack 方向仍然有价值，但 action 要改

诊断证明：

```text
cue collapse 能预测风险。
```

但实验说明：

```text
直接 freeze ReID update 不是好动作，至少当前阈值/范围不是。
```

下一步不应继续粗暴 freeze。

---

## 12. 下一步建议

### 12.1 首先不要继续 full 05

当前环境对 05 dump 处理很慢。不要在新会话一上来就反复启动 05 full diagnostic。

### 12.2 对 Trust-lite 做更细粒度 ablation

需要降低 freeze 比例：

```text
当前 freeze 比例约 28.3%，太高。
目标先降到 3%~8%。
```

建议参数：

```text
--trust-min-det-score 0.6
--trust-collapse-thresh 0.95
--trust-app-col-thresh -0.05
--trust-motion-col-thresh -0.05
```

或仅使用 collapse，不使用 app/motion col：

```text
--trust-collapse-thresh 0.95
--trust-app-col-thresh -999
--trust-motion-col-thresh -999
```

目的：

```text
只冻结最极端 cue collapse 的匹配。
```

先跑 MOT20-02 full。

### 12.3 尝试 soft update 而不是 freeze

更合理动作：

```text
高 ambiguity 时不是完全 freeze，而是 soft update：alpha = 0.97 或 0.99
```

当前 monkeypatch 版本只支持 freeze。可以扩展为：

```text
如果命中 map：
    self.update_features(new_track.curr_feat) 改成 soft alpha update
```

由于 `DMMTrack.update_features` 只接收 feat，没有 alpha 参数，需要 monkeypatch 自己实现 soft update。

理由：

```text
完全 freeze 让 memory 变旧；soft update 可能避免污染又保持适应。
```

### 12.4 不要把 cue-collapse 直接等同于 freeze

新策略应改成三档：

```text
low ambiguity:
  normal update

medium ambiguity:
  soft update

extreme ambiguity:
  freeze update
```

建议阈值：

```text
collapse < 0.80:
  normal

0.80 <= collapse < 0.95:
  soft alpha 0.97

collapse >= 0.95:
  freeze
```

### 12.5 更重要：把 cue-collapse 用于 match confidence / delay，而不是只 memory

因为当前 freeze 降低了 AssA，下一步可尝试：

```text
只在 low/mid association 或 lost reactivation 使用 cue-collapse delay，
不要动 primary high match memory。
```

但要谨慎，因为 delay 可能增加 FN。

---

## 13. 当前代码/文件清单

### 13.1 新写脚本

```text
scripts/trusttrack_rrs_diagnostic.py
scripts/trusttrack_rrs_diagnostic_lite.py
scripts/trusttrack_geo_collapse_lite.py
scripts/dmm_base_tracker_trusttrack_lite.py
outputs/analyze_rrs_csv.py
```

### 13.2 已有重要结果

```text
outputs/trusttrack_rrs_m02_norecovery.csv
outputs/analyze_rrs_m02.out
outputs/trusttrack_rrs_m03_norecovery.csv
outputs/analyze_rrs_m03.out
outputs/trust_lite_m02_full_primary/MOT20-02_summary.json
outputs/trust_lite_m02_full_primary/eval.out
outputs/trust_lite_m02_baseline_eval/eval.out
```

### 13.3 运行脚本

```text
outputs/run_trust_lite_m02_full_primary.sh
outputs/run_trust_lite_m02_300_pair.sh
outputs/run_trusttrack_rrs_m05_800.sh
```

注意：05 相关脚本容易超时，不建议立即继续。

---

## 14. 重要安全/工程注意事项

### 14.1 不要 patch 这些文件

```text
scripts/dmm_base_tracker.py
scripts/dmm_base_tracker_shadow_pda.py
```

原因：它们是 pyc wrapper 或已恢复 wrapper。

### 14.2 不要使用 apply_patch

之前有过文件被 patch fragment 覆盖的事故。后续改代码请用：

```text
python 字符串替换
write_file
新建文件
```

### 14.3 当前 dmm_base_tracker.py 是 pyc wrapper

```text
scripts/dmm_base_tracker.py
```

实际加载：

```text
outputs/recovery_backup/dmm_base_tracker.cpython-311.pyc
```

### 14.4 dmm_base_tracker_shadow_pda.py 也是 wrapper

```text
scripts/dmm_base_tracker_shadow_pda.py
```

实际加载：

```text
outputs/recovery_backup/dmm_base_tracker_shadow_pda.contact_identity.pyc
```

不要覆盖。

---

## 15. 新会话建议第一步

新会话请先做以下动作：

```bash
cd /gemini/code/FMtrack-main/FM-Track
cat docs/TRUSTTRACK_SESSION_HANDOFF_20260710.md
```

然后继续：

### Step 1：实现 TrustTrack-lite soft update

在 `scripts/dmm_base_tracker_trusttrack_lite.py` 基础上扩展：

```text
--trust-action {freeze,soft,hybrid}
--trust-soft-alpha 0.97
--trust-freeze-collapse-thresh 0.95
--trust-soft-collapse-thresh 0.80
```

### Step 2：跑 MOT20-02 full 极端少量 freeze/soft

优先组合：

```bash
python3 scripts/dmm_base_tracker_trusttrack_lite.py \
  --dump-npz outputs/alink_train_inputs/phase0_root/MOT20-02/dump_yolox_reid.npz \
  --seq MOT20-02 \
  --assoc-mode botsort_reid \
  --track-high-thresh 0.6 \
  --track-low-thresh 0.1 \
  --track-buffer 70 \
  --match-thresh 0.5 \
  --new-track-thresh 0.5 \
  --out outputs/trust_lite_m02_extreme/track_results/MOT20-02.txt \
  --summary-json outputs/trust_lite_m02_extreme/MOT20-02_summary.json \
  --trust-lite-enable \
  --trust-min-det-score 0.6 \
  --trust-collapse-thresh 0.95 \
  --trust-app-col-thresh -999 \
  --trust-motion-col-thresh -999
```

然后评估：

```bash
python3 scripts/eval_motstyle_trackeval.py \
  --benchmark-name MOT20 \
  --split-to-eval train \
  --gt-root /gemini/code/datasets/MOT20/train \
  --results-dir outputs/trust_lite_m02_extreme/track_results \
  --tracker-name trustlite02extreme \
  --work-dir outputs/trust_lite_m02_extreme/eval_work \
  --seqs MOT20-02
```

### Step 3：判断

如果 extreme freeze 仍然 HOTA/IDF1 下降：

```text
memory freeze 这条线先暂停。
```

转向：

```text
cue-collapse 作为 confidence/delay 或 tracklet-conditioned detection boost，而不是 memory update gate。
```

---

## 16. 当前最终判断

```text
TrustTrack 诊断方向成立：cue collapse 是强风险信号。
但当前 action=freeze memory 的第一版效果不好：MOT20-02 HOTA 从 68.748 降到 68.047。
所以下一步不是否定 TrustTrack，而是反思 action：
  freeze 太粗暴，应改为 soft/hybrid 或用于匹配置信度/延迟，而不是大量冻结 ReID memory。
```

最重要的一句话：

```text
信号成立，不代表动作正确。
cue collapse 能预测风险；但 memory freeze 不是当前最优干预。
```
