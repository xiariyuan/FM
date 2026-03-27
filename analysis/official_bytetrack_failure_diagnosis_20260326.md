# Official ByteTrack 诊断驱动分析

日期：2026-03-26

本报告只回答三个问题：

1. strict official ByteTrack 的第一阶段高分关联，真实失败模式是什么。
2. 这些失败里，有多少真正落在 current local-conflict 插件的作用域内。
3. 下一步最小改动应该先打在哪里，而不是先发明一个新模块。

本次诊断产物由新脚本生成：

- `scripts/analyze_official_bytetrack_failure_slices.py`
- `analysis/official_bytetrack_failure_diagnosis_20260326/summary.json`
- `analysis/official_bytetrack_failure_diagnosis_20260326/per_sequence.csv`
- `analysis/official_bytetrack_failure_diagnosis_20260326/slice_summary.csv`
- `analysis/official_bytetrack_failure_diagnosis_20260326/cluster_coverage.csv`
- `analysis/official_bytetrack_failure_diagnosis_20260326/teacher_alignment.csv`
- `analysis/official_bytetrack_failure_diagnosis_20260326/recoverable_coverage_reason_summary.csv`

## 1. 数据流边界先钉死

当前 canonical 载体仍然是 strict official ByteTrack shared-detection path，不是内部 host。

数据流如下：

1. `third_party/ByteTrack/yolox/tracker/byte_tracker.py`
   official ByteTrack 先把检测框按 `track_thresh` 分成高分和低分，再在高分框上做 first-stage association；匹配代价是 IoU 距离，再经 `matching.fuse_score` 融合 detection score，最后走 Hungarian。

2. `third_party/ByteTrack/yolox/tracker/byte_tracker_local_conflict.py`
   插件只包 official ByteTrack 第一阶段高分关联前的一小段。它读取第一阶段的 IoU/refined score、几何、gap/history，构建局部 bipartite components，筛掉不满足 size 条件的 cluster，然后才把这些 cluster 送进 learned operator。

3. `scripts/build_runtime_assoc_replay_labels.py`
   replay 标注把每个 detection row 的 top-k candidate dump 回放到 GT，产出 group-level 的 `rank_margin`、`rank_entropy`、`rank_top1_correct`、`group_is_ambiguous`、`group_is_recoverable` 等字段。也就是说，我们现在已经有足够信息直接看 baseline 在什么 row 上错。

4. `scripts/build_local_conflict_set_predictor_dataset.py`
   当前训练数据不是直接从全部 replay rows 学，而是先把 row 用共享 top-k candidate track 连成局部冲突图，再用 `min_detections=2 / max_detections=8 / max_tracks=32` 做过滤，最后才得到 `cluster_examples.jsonl`。

因此，baseline failure 分析和插件 coverage 分析必须分开做：

- baseline failure：看 `labeled_replay_top8.groups.jsonl`
- plugin coverage：看 `cluster_examples.jsonl`

## 2. baseline 真正怎么失败

先看第一阶段高分关联本身。

全体 first-stage 正样本 group 一共有 `46050` 个，但 `rank_top1_correct=0` 的真实 top1 失败只有 `145` 个，错误率仅 `0.3149%`。

这件事已经说明两个结论：

- official ByteTrack first-stage 在这个 shared-detection protocol 下已经非常强。
- 论文增益不可能来自“普遍修正大量第一阶段错误”，而只能来自对极少数高价值失败 slice 的高精度干预。

更关键的是，这 `145` 个失败不是散布在全空间里，而是几乎完全集中在少量高歧义样本：

- `group_is_ambiguous=1` 的 group 一共 `286` 个，其中 `145` 个 top1 错，错误率 `50.7%`
- `group_is_ambiguous=0` 的正样本 group 一共 `45764` 个，top1 错 `0`

换句话说，当前 official ByteTrack 的 first-stage 缺陷几乎可以收缩成一句话：

**不是普遍排序失败，而是少量 ambiguous rows 的排序失败。**

进一步看 slice：

- 最低 `rank_margin` 五分位吃掉了 `137/145 = 94.5%` 的全部错误
- 最高 `rank_entropy` 五分位吃掉了 `129/145 = 89.0%` 的全部错误
- 最低 `positive_refined_score` 五分位吃掉了 `145/145 = 100%` 的全部错误
- 最短 `positive_hist_len` 五分位吃掉了 `97/145 = 66.9%` 的全部错误

这说明 first-stage 的主要失败接口不是 long-gap，不是全局信息缺失，也不是大面积几何崩坏，而是：

- 排名前几名候选极近，margin 很小
- row 内分布熵高，不确定性高
- 正确轨迹本身得分偏低
- 更偏向短历史、弱稳定性的轨迹

按序列看，错误也明显集中在拥挤场景：

- `MOT17-05-FRCNN`: `54` 个错误，正样本 top1 错误率 `2.24%`
- `MOT17-10-FRCNN`: `45` 个错误，错误率 `1.39%`
- `MOT17-13-FRCNN`: `21` 个错误，错误率 `0.45%`
- `MOT17-04-FRCNN`: 只有 `1` 个错误

所以 baseline defect 的核心不是“ByteTrack 普遍不行”，而是：

**在 crowd-heavy sequence 上，first-stage 的少量 ambiguous rows 会错。**

## 3. 插件现在到底覆盖了多少真实错误

当前 `edit_utility` 训练集来自三个 train-half 序列：

- `MOT17-05-FRCNN`
- `MOT17-09-FRCNN`
- `MOT17-11-FRCNN`

在这三个序列里，replay 标注出的真实 recoverable first-stage 错误一共 `68` 个。

但 current cluster builder 只把其中 `33` 个纳入了 `cluster_examples.jsonl`，覆盖率只有 `48.5%`。

这一步非常关键，因为它说明：

**在 teacher 宽窄之前，current plugin scope 自己就已经漏掉了一半真实错误。**

更重要的是，这些漏掉的错误不是因为 singleton，不是因为 `min_detections=2` 太严，而是几乎全部被 size filter 挡掉：

- `eligible`: `33`
- `skipped_large`: `35`
- `singleton_small`: `0`

并且两类 component 的规模差异很明显：

- 被纳入的 recoverable rows：平均 `7.09` detections / `10.21` tracks
- 被过滤掉的 recoverable rows：平均 `10.71` detections / `13.89` tracks

这说明真实错误更常出现在更拥挤、更大的 local conflict component 中，而 current `max_detections=8 / max_tracks=32` 恰好把这部分切掉了。

因此，coverage 问题的主矛盾不是 “top-k 太小导致没连上”，也不是 “冲突图只抓到少量单行”，而是：

**真实高价值失败往往就在 large crowded component 里，而 current size cap 先把它们扔掉了。**

## 4. current teacher 为什么会一会儿太开，一会儿又太关

这个问题也可以直接从 cluster diagnostics 里看到。

在当前 `edit_utility` 训练集上：

- `cluster_should_intervene_edit=1` 的正样本 cluster 有 `113` 个
- 其中真正含有 top1 error row 的只有 `25` 个
- cluster-level precision 只有 `22.1%`

也就是说，`edit_utility` 的正样本定义仍然太宽。它并不是在说“这里真的有 baseline 排序错误”，而是在说“这里存在 oracle 与 host 的编辑差异”。这两者不是一回事。

但 `cluster_should_intervene_soft` 又走到了另一头：

- 正样本 cluster 只有 `22` 个
- 这 `22` 个全部都命中了真实 error cluster，precision `100%`
- 但它只覆盖了 `29` 个 error clusters 里的 `22` 个

所以现状不是 family 不行，而是 supervision 已经被拉成两个极端：

- `edit_utility`: coverage 较大，但 precision 太差
- `soft/rescue`: precision 极高，但 density 太低，很容易学成 near no-op

这和我们已经看到的在线现象是一致的：

- 太宽的 teacher 更容易把插件推向 harmful intervention
- 太稀的 teacher 更容易把插件推向 zero-coverage / all-defer

## 5. 这轮诊断给出的最小改动方向

这里先不给新模块，只给最小改动顺序。

### 第一优先级：先修 plugin coverage，不是先换 backbone

当前最先该动的不是 `set_predictor_v2` 结构，而是 local conflict scope。

原因很直接：

- baseline 真错误几乎都在少量 ambiguous rows
- train split 里真实 recoverable rows 有 `68`
- current cluster builder 只吃到 `33`
- 丢掉的 `35` 个几乎全是 `skipped_large`

所以第一步应该优先解决：

- large component 怎么保留
- 不是直接把 `max_detections` 盲目放大，而是设计一个 **crowded component 的保留策略**

最小可尝试方向：

1. 对 `num_detections > 8` 的 component，不直接丢弃，而是围绕 ambiguous / low-margin rows 做局部裁剪子图。
2. 或者对 large component 做二次拆分，而不是全簇跳过。
3. 在诊断脚本里继续跟踪 `recoverable_groups_covered`，把它当成 stage0 必看指标。

### 第二优先级：teacher 要更贴近“真实 baseline 错误”，而不是宽泛 edit

当前 `edit_utility` 正样本 precision 只有 `22.1%`，这是太宽。

但 `soft/rescue` 又太稀。

因此下一轮 teacher 不应再回答“oracle 和 host 是否有编辑差异”，而应该更贴近：

- row 是否真的处于 ambiguous / recoverable 区域
- cluster 是否真的包含 baseline first-stage 错误
- large crowded component 中，这个子图干预是否有净价值

也就是说，teacher 应以 **真实失败 slice** 为锚，而不是以“任何 host-vs-oracle edit” 为锚。

### 第三优先级：评测也要跟着 failure slice 走

后面不应该只看总 HOTA，而应该固定跟踪三类前置诊断：

1. first-stage recoverable row coverage
2. crowded large-component coverage
3. intervention precision in ambiguous clusters

如果这三项不先变好，再去赌 end-to-end paired gain，基本还是空跑。

## 6. 当前可以收束成的结论

一句话版本：

**official ByteTrack 的 baseline 缺陷不是普遍关联错误，而是 crowd-heavy 场景中少量 ambiguous first-stage rows 的排序失败；current plugin 甚至还没充分进入这个失败区域，因为现有 cluster size cap 先挡掉了约一半 train-split 的真实 recoverable rows。**

所以接下来最值钱的工作顺序是：

1. 先修 crowded large-component coverage
2. 再把 teacher 锚到真实失败 slice
3. 最后才讨论模块结构要不要增强

不是先想一个更强模块去赌它会涨点。
