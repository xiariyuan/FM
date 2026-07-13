# TrustTrack / ShadowPDA 实验交接文档

> 写给新会话读取使用。当前工作目录：`/gemini/code/FMtrack-main/FM-Track`
>
> 重要原则：**不要修改 `scripts/dmm_base_tracker.py` 和已有稳定 wrapper；不要使用 apply_patch 覆盖核心文件。** 本轮新增脚本均为独立文件或 outputs 下运行脚本。

---

## 1. 当前总体结论

本轮会话从 ShadowPDA-V3 继续推进到 TrustTrack 诊断与 TrustTrack-lite memory gate。核心结论已经更新：

```text
不要继续把主线押在 ShadowPDA low public recovery 上。
ShadowPDA-V3 只能作为低分证据 fallback。
真正应该主攻的是主关联阶段的 AssA / IDF1 / IDSW。
```

但 TrustTrack 原始命题也要修正：

```text
原命题：线索可靠性反转 RRS 导致错误关联。
诊断结果：RRS 本身不是稳定单调风险信号。
更强信号：cue collapse / max_pair_margin / app_col_margin / motion_col_margin。
```

更准确的论文命题：

```text
Association failures are better predicted by local cue ambiguity collapse
than by raw cue similarity or fixed cue reliability reversal.
```

中文：

```text
关联失败更直接地由局部线索区分力塌陷预测，
而不是由某条线索相似度高低或简单线索可靠性反转预测。
```

---

## 2. 用户目标与判断

用户最初提到之前废弃 idea 的 leaderboard HOTA 约 63.6，后续明确希望：

```text
指标越高越好，至少希望 HOTA 80+，更高更好。
```

需要区分：

```text
官方 MOT20 test leaderboard：榜首 HOTA 约 67.x，HOTA 80+ 不现实。
内部 train / 自定义验证集：如果要 HOTA 80+，必须系统级提升 detection + association。
```

从用户贴的榜单看，Rank 29 与 Top public 方法的核心差距主要是：

```text
DetA 不一定差很多；
AssA / IDF1 / IDSW 差很多。
```

所以短期要冲 64/65 或更高：

```text
优先攻主关联、ID consistency、ReID memory 防污染。
```

长期更高：

```text
还要做 detection oracle、detector postprocess、box quality / visible-full box / learned association。
```

---

## 3. 已完成的诊断：MOT20-02 / MOT20-03

### 3.1 新增诊断脚本

已写入：

```text
scripts/trusttrack_rrs_diagnostic.py
outputs/analyze_rrs_csv.py
```

作用：

```text
离线重构已有 tracking result + detection dump + GT，
计算 app / motion / IoU / shape cue 的 row_margin / col_margin / pair_margin，
并统计 rrs / cue_collapse 等信号是否预测 bad match。
```

注意：当前是离线重构，不是真实 tracker 内部 cost matrix。后续论文级严谨性需要在真实 tracker 内部 logger 中复核。

---

### 3.2 MOT20-02 诊断结果

输入：

```text
outputs/alink_train_inputs/phase0_root/MOT20-02/dump_yolox_reid.npz
outputs/shadow_pda_m02_full_norecovery/track_results/MOT20-02.txt
```

输出：

```text
outputs/trusttrack_rrs_m02_norecovery.csv
outputs/trusttrack_rrs_m02_norecovery.out
outputs/analyze_rrs_m02.out
```

统计：

```text
rows = 145060
class_counts = {
  correct: 138672,
  det_fp: 1615,
  wrong_id: 4715,
  unknown: 58
}
```

原始 RRS 不单调：

```text
rrs=0: bad_rate ≈ 7.32%
rrs 0.193~0.358: bad_rate ≈ 5.41%
rrs 0.358~0.434: bad_rate ≈ 3.24%
rrs 0.434~0.555: bad_rate ≈ 2.34%
rrs 0.555~0.893: bad_rate ≈ 5.51%
```

cue_collapse 很强：

```text
collapse 0.001~0.206: bad_rate = 0.45%
collapse 0.524~0.675: bad_rate = 7.28%
collapse 0.784~0.843: bad_rate = 16.43%
collapse 0.843~0.951: bad_rate = 26.44%
collapse 0.951~1.335: bad_rate = 51.14%
```

app_col_margin 很强：

```text
app_col_margin lowest 10%:
bad_rate = 26.94%
wrong/FP rate = 18.13%
```

motion_col_margin 也强：

```text
motion_col_margin lowest 10%:
bad_rate = 19.58%
wrong/FP rate = 13.72%
```

结论：MOT20-02 支持 cue collapse / col-margin ambiguity，不支持原始 RRS 作为主信号。

---

### 3.3 MOT20-03 诊断结果

输出：

```text
outputs/trusttrack_rrs_m03_norecovery.csv
outputs/trusttrack_rrs_m03_norecovery.out
outputs/analyze_rrs_m03.out
```

统计：

```text
rows = 302553
bad = 7733
wrongfp = 5619
```

RRS 仍不单调：

```text
rrs=0: bad_rate ≈ 2.60%
rrs 0.313~0.449: bad_rate ≈ 1.17%
rrs 0.449~0.519: bad_rate ≈ 0.87%
rrs 0.519~0.638: bad_rate ≈ 1.15%
rrs 0.638~0.978: bad_rate ≈ 2.61%
```

cue_collapse 仍然强：

```text
collapse 0.001~0.116: bad_rate = 0.48%
collapse 0.489~0.642: bad_rate = 3.49%
collapse 0.642~0.732: bad_rate = 9.22%
collapse 0.732~0.884: bad_rate = 16.40%
collapse 0.884~1.343: bad_rate = 34.70%
```

app_col_margin：

```text
lowest 10%:
bad_rate = 14.50%
wrong/FP rate = 10.86%
```

motion_col_margin：

```text
lowest 10%:
bad_rate = 14.04%
wrong/FP rate = 10.45%
```

结论：MOT20-03 与 MOT20-02 高度一致。

---

## 4. MOT20-05 诊断状态

05 full 诊断未完成。

检查结果：

```text
outputs/trusttrack_rrs_m05_norecovery.csv  不存在
outputs/trusttrack_rrs_m05_norecovery.out  存在但 size=0
```

后来写了轻量版脚本，但 05 dump 读取和 I/O 很慢，未成功产出。

新增但尚未成功产出的脚本：

```text
scripts/trusttrack_rrs_diagnostic_lite.py
scripts/trusttrack_geo_collapse_lite.py
outputs/run_trusttrack_rrs_m05_800.sh
```

建议下一会话：不要继续盲跑 full 05；先预裁剪 MOT20-05 的前 300/800 帧小 npz，再跑轻量诊断。

---

## 5. TrustTrack-lite memory gate 实现

### 5.1 新增脚本

已写入：

```text
scripts/dmm_base_tracker_trusttrack_lite.py
```

设计目标：

```text
最小干预版本：
- 不改变 Hungarian matching
- 不改变 Kalman update
- 不改变 detection output
- 不改变 track lifecycle
- 只在高 ambiguity / cue collapse 匹配上冻结 ReID feature update
```

核心思想：

```text
High-ambiguity matches may be acceptable for short-term continuity,
but should not contaminate long-term identity memory.
```

中文：

```text
高歧义匹配可以暂时维持短期轨迹连续，
但不应该污染长期身份记忆。
```

---

### 5.2 实现方式

`DMMTrack.update` 的字节码行为已确认：

```text
1. 更新 frame_id / tracklet_len
2. 更新 Kalman mean/covariance
3. 如果 new_track.curr_feat 不为空，则 update_features(new_track.curr_feat)
4. 更新 state / score / det_global_idx
```

脚本通过 monkeypatch：

```text
如果 (track_id, det_global_idx) 命中 freeze map：
    临时把 new_track.curr_feat 置为 None
    调用原始 DMMTrack.update()
    再恢复 curr_feat
```

因此：

```text
box/Kalman/score/det_global_idx 都照常更新；
只有 ReID memory update 被跳过。
```

---

### 5.3 Trust / collapse 定义

每帧 `tracker.update()` 前，用当前 active tracks 和当前 detections 计算：

```text
app / motion / IoU / shape cue matrices
```

定义：

```text
row_margin_c = track 对所有 detections 的 top1-top2 区分力
col_margin_c = detection 对所有 tracks 的 top1-top2 区分力
pair_margin_c = min(row_margin_c, col_margin_c)
trust = max_c pair_margin_c
collapse = 1 - trust
```

freeze 条件：

```text
collapse >= trust_collapse_thresh
or app_col_margin <= trust_app_col_thresh
or motion_col_margin <= trust_motion_col_thresh
```

当前 primary-only 参数：

```text
--trust-min-det-score 0.6
--trust-collapse-thresh 0.85
--trust-app-col-thresh 0.05
--trust-motion-col-thresh 0.05
```

---

## 6. TrustTrack-lite 实验结果：MOT20-02 full primary-only

### 6.1 运行脚本

已写入：

```text
outputs/run_trust_lite_m02_full_primary.sh
```

输出目录：

```text
outputs/trust_lite_m02_full_primary/
```

结果文件：

```text
outputs/trust_lite_m02_full_primary/MOT20-02_summary.json
outputs/trust_lite_m02_full_primary/eval.out
outputs/trust_lite_m02_full_primary/track_results/MOT20-02.txt
```

---

### 6.2 与 no-recovery baseline 的严格对照

Baseline eval 目录：

```text
outputs/trust_lite_m02_baseline_eval/eval.out
```

Baseline 使用：

```text
outputs/shadow_pda_m02_full_norecovery/track_results/MOT20-02.txt
```

#### Baseline MOT20-02 no-recovery

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

#### TrustTrack-lite primary-only memory freeze

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

Trust stats：

```text
candidate_pairs         = 8,225,821
freeze_pairs_predicted  = 8,125,204
feature_updates_frozen  = 41,174
feature_updates_normal  = 104,358
frames                  = 2,782
```

### 6.3 关键分析

TrustTrack-lite primary-only 的现象：

```text
IDSW: 428 -> 419  改善 9
FN:   11556 -> 11521 改善 35
FP:   2329 -> 2313 改善 16
Frag: 765 -> 759 改善 6
IDs:  475 -> 466 改善 9
```

但：

```text
AssA: 60.053 -> 58.816 下降 1.237
IDF1: 75.205 -> 74.723 下降 0.482
HOTA: 68.748 -> 68.047 下降 0.701
```

结论：

```text
Memory freeze 过强/过宽，虽然减少了一些 IDSW、FN、FP、Frag，
但破坏了身份关联连续性，导致 AssA/IDF1/HOTA 明显下降。
```

不要把当前 primary-only memory freeze 当成功结果。它是一个重要 negative result。

---

## 7. 已确认的实现/计算问题

### 7.1 外部重算 cue matrix 很慢

当前 TrustTrack-lite 在 wrapper 外部每帧重算：

```text
tracks × detections × 2048-d appearance matrix
```

即使加了 `--trust-min-det-score 0.6`，full 02 能跑完，但比较慢。

最终更合理做法：

```text
在 tracker 内部真实 association cost matrix 上直接计算 row/col margin，
不要外部重算所有 cue。
```

但当前 `DMMBaseTracker` 来自 pyc，难以直接编辑内部 association 阶段。

### 7.2 外部 BoT-SORT 里已有 SPOT 逻辑，但当前 DMM wrapper 不透传

检查过：

```text
external/BoT-SORT-main/tracker/bot_sort.py
```

其中已有：

```text
spot_enable
spot_freeze_app
spot_margin_thresh
spot_row_margin
spot_col_margin
spot_margin
```

但当前 `scripts/dmm_base_tracker.py` 是 pyc wrapper，`TrackerConfig` 不含 SPOT 字段，所以不能简单通过 CLI 打开。

---

## 8. 重要反思

### 8.1 TrustTrack 方向仍然成立，但第一版 action 错了

诊断证明：

```text
cue collapse / col-margin ambiguity 强预测坏匹配。
```

但直接做 memory freeze 并不是自动正向。

可能原因：

```text
1. high-ambiguity 匹配未必是错匹配；冻结太多正常更新，导致 ReID memory 老化。
2. freeze 影响后续 appearance matching，降低 AssA。
3. 当前 gate 是预计算所有 track-det pair，不一定等同真实 matched pair 的内部风险。
4. freeze 条件过宽，实际冻结 41k 次 feature updates，可能过强。
5. HOTA/AssA 对 identity continuity 更敏感，单纯减少 IDSW 不够。
```

### 8.2 下一步不能继续盲目扩大 freeze

当前结果明确：

```text
primary-only freeze 已经让 HOTA -0.701。
```

所以不要再跑更宽松阈值，例如：

```text
collapse_thresh 0.80
app_col_thresh 0.08
motion_col_thresh 0.08
```

那大概率更差。

---

## 9. 建议下一步

### 9.1 首先做 freeze ablation 缩窄阈值

当前 freeze 太多。下一步应该极端保守，只冻结最危险 1%~3% 更新，验证是否能保留 IDSW 改善而避免 AssA 下降。

建议 sweep：

```text
A: --trust-collapse-thresh 0.95 --trust-app-col-thresh -1.0 --trust-motion-col-thresh -1.0
B: --trust-collapse-thresh 0.98 --trust-app-col-thresh -1.0 --trust-motion-col-thresh -1.0
C: --trust-collapse-thresh 1.00 --trust-app-col-thresh -1.0 --trust-motion-col-thresh -1.0
D: disable collapse, only app_col very low:
   --trust-collapse-thresh 999 --trust-app-col-thresh -0.05 --trust-motion-col-thresh -1.0
E: disable collapse, only motion_col very low:
   --trust-collapse-thresh 999 --trust-app-col-thresh -1.0 --trust-motion-col-thresh -0.05
```

注意：脚本当前阈值逻辑是：

```text
collapse >= threshold 触发
app_col <= threshold 触发
motion_col <= threshold 触发
```

如果要关闭某项：

```text
collapse 关闭可设 999
app_col/motion_col 关闭可设 -999 或很小值
```

目标不是马上提 HOTA，而是找出：

```text
feature_updates_frozen 占比降到 1%~3% 后，AssA 是否仍下降。
```

---

### 9.2 更好的 action：不要 freeze，而是 soft update

当前 binary freeze 太硬。更合理：

```text
高 ambiguity 时不是完全不更新，
而是使用 conservative EMA：alpha 更高，例如 0.98 / 0.99。
```

但是 `DMMTrack.update_features` 当前只接受 feat，不支持 mode。`DMMTrack.update` pyc 中直接调用 `self.update_features(new_track.curr_feat)`。

可通过 monkeypatch `DMMTrack.update_features` 或 `DMMTrack.update` 实现 soft：

```text
如果 key 命中 soft map：
    临时提高 self.alpha 到 0.98/0.99
    调用原 update_features
    恢复 self.alpha
```

下一步可以新建：

```text
scripts/dmm_base_tracker_trusttrack_soft.py
```

或者在现有 `dmm_base_tracker_trusttrack_lite.py` 加：

```text
--trust-action freeze|soft
--trust-soft-alpha 0.98
```

预期 soft 比 freeze 更合理，避免 memory 老化。

---

### 9.3 更根本：用 cue-collapse 做 delayed matching，而不是 memory freeze

诊断表明 cue collapse 预测坏匹配，但 action 可能应该是：

```text
高 collapse 且匹配不强时，不要提交匹配 / 延迟一帧，
而不是提交匹配但冻结 memory。
```

但是 delayed matching 会影响 FN/Frag，需要在真实 association 内部做，当前 pyc wrapper 难实现。

短期仍可做 soft update；中期需要可编辑 tracker 源码。

---

### 9.4 仍需做 oracle

必须补：

```text
1. current-output ID oracle
2. detection dump oracle
```

目的：判断如果只修 ID assignment，HOTA ceiling 能到多少；如果 detection oracle 也不够，就必须攻 detector/postprocess。

---

## 10. 下一会话建议的第一条指令

建议让新会话直接读本文档，然后执行：

```text
请读取 docs/TRUSTTRACK_HANDOFF_NEXT_SESSION.md，
继续从第 9 节开始，先实现/运行 TrustTrack-lite 的保守阈值 sweep，
不要扩大 freeze，不要继续跑 05 full 诊断。
```

优先执行的短命令/脚本思路：

```text
1. 基于 outputs/run_trust_lite_m02_full_primary.sh 复制出 sweep 脚本。
2. 每次只改 trust thresholds。
3. 每个配置跑 MOT20-02 full 并 TrackEval。
4. 记录：HOTA / AssA / IDF1 / IDSW / FN / FP / Frag / feature_updates_frozen。
5. 如果所有 freeze 配置都伤 HOTA，则转向 soft update。
```

---

## 11. 文件清单

### 诊断相关

```text
scripts/trusttrack_rrs_diagnostic.py
scripts/trusttrack_rrs_diagnostic_lite.py
scripts/trusttrack_geo_collapse_lite.py
outputs/analyze_rrs_csv.py
outputs/trusttrack_rrs_m02_norecovery.csv
outputs/trusttrack_rrs_m02_norecovery.out
outputs/analyze_rrs_m02.out
outputs/trusttrack_rrs_m03_norecovery.csv
outputs/trusttrack_rrs_m03_norecovery.out
outputs/analyze_rrs_m03.out
outputs/run_trusttrack_rrs_m05_800.sh
```

### TrustTrack-lite 实现与运行

```text
scripts/dmm_base_tracker_trusttrack_lite.py
outputs/run_trust_lite_m02_full_primary.sh
outputs/run_trust_lite_m02_300_pair.sh
outputs/trust_lite_m02_full_primary/
outputs/trust_lite_m02_baseline_eval/
outputs/trust_lite_m02_120_notrust/
outputs/trust_lite_m02_120_primary/
outputs/trust_lite_m02_300_notrust/
outputs/trust_lite_m02_300_primary/
```

### 重要 baseline

```text
outputs/shadow_pda_m02_full_norecovery/track_results/MOT20-02.txt
outputs/shadow_pda_v3_m02_full/track_results/MOT20-02.txt
outputs/shadow_pda_v3_precision_m02_full/track_results/MOT20-02.txt
```

---

## 12. 最终当前状态一句话

```text
TrustTrack 的诊断信号是成立的：cue collapse / col-margin ambiguity 强预测坏匹配；
但第一版 binary memory freeze 是负结果：MOT20-02 HOTA 从 68.748 降到 68.047。
下一步不要扩大 freeze，而要做极保守 freeze sweep 或改成 soft update。
```
