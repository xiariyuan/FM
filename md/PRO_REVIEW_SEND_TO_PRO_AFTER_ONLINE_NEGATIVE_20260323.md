# Send To Pro: After Online Negative Result (2026-03-23)

这是一份针对最新 online negative signal 的发送版提示词。目标不是重新开放所有旧方向，而是基于最新结果直接做 go/kill 决策。

```markdown
请你把自己当成独立研究顾问，只审下面这条已经收敛下来的 MOT 主线。不要重新发散到旧方向，也不要泛泛给建议。请基于我给出的完整历史、最新 offline/online 证据，直接判断这条 learned online controller 主线现在是继续、收缩，还是停止。

## 1. 项目背景

我当前做的是 tracking-by-detection 范式下的 MOT 研究。
当前固定宿主是 `base_reid_da`，但它不是主角方法本身，只是当前最稳定的 host baseline。

当前主角方法是：
- `competition-aware association controller`
- 目标是在 frozen strong host 上，只对 runtime local conflict sets 做 selective intervention
- 第一阶段只做 `primary association` 上的 `keep / rerank / null`
- 当前实际 online 接回时，为了降低风险，先只做 `rerank-first`

注意：
- 我当前不是在改 detector
- 也不是在重写整个 tracker
- 我现在关心的是：这个 learned online controller 主线，是否还能继续

## 2. 已明确停掉或降级的路线

### 已停主线
- Laplace / MTCR / HACA / pairwise residual safe plugin
- 当前 frequency-aware / spatial-freq interaction 主线

### 已降级为二线
- runtime replay safe plugin

因此，请你不要建议我回到这些路线当主线，除非你能基于下面事实明确指出为什么必须复活。

## 3. 当前 host / baseline 控制链

这是 MOT17 proxy0213（MOT17-02 / 13）上的控制链：

### `base_spatial`
- HOTA = 52.233
- AssA = 43.089
- IDF1 = 58.557
- MOTA = 72.382
- IDSW = 1148

### `base_reid_da`
- HOTA = 52.704
- AssA = 43.997
- IDF1 = 58.608
- MOTA = 73.361
- IDSW = 834

### `full_reid_da`
- HOTA = 51.733
- AssA = 42.365
- IDF1 = 57.969
- MOTA = 73.471
- IDSW = 791

结论：
- `base_reid_da` 相对 `base_spatial` 有真实净提升
- `full_reid_da` 已经给出负信号，所以 frequency family 主线已停
- 当前固定 `base_reid_da` 作为 host

## 4. 当前主线已经跑通的 offline 事实

### 4.1 runtime conflict-set 数据已经成功构建

proxy0213 summary:
- groups = 29319
- positive_groups = 19756
- recoverable_groups = 4751
- recoverable_rate_among_positive = 0.24048
- bridge_rate_among_positive = 0.03801

这说明：
- hard conflict sets 数量是够的
- 可恢复重排组约占 positive groups 的 24%
- continuity bridge 比例很小，所以第一阶段先做 rerank 是合理的

### 4.2 stage1 offline controller 已经能学到 signal

当前正式 offline 结果：
- best_epoch = 10
- val_action_acc = 0.618739
- val_rerank_candidate_acc = 0.915464
- val_rerank_action_acc = 0.657732
- status = ok

这说明：
- 离线 candidate winner prediction 很强
- 但 action head 只算中等
- continuity/bridge 头当前不强

## 5. 最新 online integration 结果

我已经把 controller 接回 `base_reid_da` host 的 primary association 路径。
注入方式是：
- `pre-Hungarian`
- `primary association only`
- 不碰 secondary/newborn/lifecycle
- online 只用训练时那套 8 个 observed group features 和 6 个 candidate features

### 5.1 `noop` baseline

这不是旧 host 表，而是同一套 online harness 下的 no-op 接回基线：

- HOTA = 52.758
- AssA = 44.038
- IDF1 = 58.276
- MOTA = 73.232
- IDSW = 847

### 5.2 `rerank_only` online

这版用的是较平滑的 local residual 重排：

- HOTA = 52.608
- AssA = 43.737
- IDF1 = 58.135
- MOTA = 73.179
- IDSW = 855

相对 `noop`：
- HOTA -0.150
- AssA -0.301
- IDF1 -0.141
- MOTA -0.053
- IDSW +8

### 5.3 `rerank_minimal` online

这版按更保守的 operator 重跑：
- hard trigger
- minimal winner override
- 只打 margin < 0.10 的低 margin 冲突组

结果：
- HOTA = 52.580
- AssA = 43.724
- IDF1 = 58.198
- MOTA = 73.219
- IDSW = 849

相对 `noop`：
- HOTA -0.178
- AssA -0.314
- IDF1 -0.078
- MOTA -0.013
- IDSW +2

结论：
- 两个 learned online 版本都没有打过 `noop`
- 当前最硬的负信号，不是绝对值 52.x，而是同一 harness 下 learned online controller 没超过 no-op baseline

## 6. 我当前的理解

我现在认为问题已经不是：
- detector 不够强
- frequency 模块少一点点
- 再多训几轮 controller 就会自然变好

而更像是：
- offline 学到的 candidate-level signal，没有成功转成 online gain
- objective / trigger / online operator / Hungarian coupling 之间仍然存在 mismatch

## 7. 这次请你只回答 go/kill 决策问题

请你直接回答下面这些问题，不要泛泛谈可能性：

1. 基于现在这组证据，这条 learned online controller 主线是否还值得继续？请给出明确判断：`GO / NARROW GO / KILL`。

2. 如果不是直接 kill，那么接下来信息增益最高的唯一实验应该是什么？
   我当前能想到的候选是：
   - `oracle rerank upper bound`
   - 换更严格的 held-out validation（seq-held-out / time-block holdout）
   - 改 loss / trigger / operator
   - 直接转向更结构化的 competition / graph / tracklet-level decision

3. 以你看，这两个 online negative signal 更像：
   - controller 没学到真正可用的 online signal
   - online operator / trigger 不对
   - label / objective mismatch
   - host 上剩余 headroom 本来就太小
   请给排序。

4. 你是否支持我现在继续做 `oracle rerank upper bound`？
   如果支持，请明确说它为什么是最高信息增益实验。
   如果不支持，请给出更优替代。

5. 如果 `oracle rerank upper bound` 也不明显高于 `noop`，你是否建议立刻停止这条 learned online controller 主线？

6. 如果这条线应当停止，你建议我下一跳是：
   - competition-aware graph / local conflict graph
   - tracklet-level continuity / stitching
   - 其他更结构化的 association decision unit
   请给出你认为最理性的唯一主方向。

## 8. 我随附的材料

你可以结合下面几份压缩包一起判断：

- offline stage1 bundle：
  - `outputs/competition_assoc_stage1_bundle_20260323_093230.zip`
- proxy conflict-case bundle：
  - `outputs/competition_assoc_proxy_bundle_20260323_093231.zip`
- online noop bundle：
  - `outputs/competition_assoc_online_bundle_20260323_095538.zip`
- online rerank_only bundle：
  - `outputs/competition_assoc_online_bundle_20260323_113623.zip`
- online rerank_minimal bundle：
  - `outputs/competition_assoc_online_bundle_20260323_114949.zip`

请你不要只重复“继续试试”或“也许有潜力”。我现在需要的是：基于这些最新负结果，直接做未来 1-2 步实验的 go/kill 决策。
```
