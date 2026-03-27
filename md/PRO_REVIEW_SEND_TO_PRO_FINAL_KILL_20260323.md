# Send To Pro: Final Kill Decision And Next Mainline (2026-03-23)

这是一份在 `oracle_rerank` 结果出来之后的最终发送版提示词。用途不是再讨论当前 row-local rerank 要不要微调，而是请对方确认 kill，并审下一主线该切到哪。

```markdown
请你把自己当成独立研究顾问，只审下面这条已经收敛下来的 MOT 主线。不要重新发散到旧方向，也不要泛泛谈可能性。我现在需要你基于完整历史和最新 oracle 结果，直接判断：

1. 当前 `row-local learned online controller` 是否已经可以判定为 kill
2. 如果 kill，下一主线应该切到哪一种更大的 decision unit

## 1. 当前项目背景

我当前做的是 tracking-by-detection 范式下的 MOT 研究。
当前固定 host 是 `base_reid_da`，它不是主角方法本身，只是当前稳定的宿主。

过去这一阶段我研究的是：
- 在 frozen host 上学习一个 `competition-aware association controller`
- 第一阶段只做 `primary association`
- 只做 runtime local conflict set 上的 `keep / rerank / null`
- 实际上线时，先只做 `rerank-first`

## 2. 已明确停掉或降级的旧路线

### 已停主线
- Laplace / MTCR / HACA / pairwise residual safe plugin
- 当前 frequency-aware / spatial-freq interaction 主线

### 已降级为二线
- runtime replay safe plugin

因此请你不要再建议我回到这些路线当主线，除非你能基于下面事实明确指出为什么必须复活。

## 3. 当前 host / baseline 控制链

这是 MOT17 proxy0213（MOT17-02 / 13）上的关键控制链：

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

## 4. 当前 row-local controller 的 offline 事实

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
- continuity bridge 占比很小，所以第一阶段先做 rerank 是合理的

### 4.2 stage1 offline controller 学到了 candidate-level signal

当前正式 offline 结果：
- best_epoch = 10
- val_action_acc = 0.618739
- val_rerank_candidate_acc = 0.915464
- val_rerank_action_acc = 0.657732
- status = ok

## 5. 当前 online integration 的完整结果

我已经把 controller 接回 `base_reid_da` host 的 `primary association` 路径。
注入方式是：
- `pre-Hungarian`
- `primary association only`
- 不碰 secondary/newborn/lifecycle

### 5.1 `noop`

同一 harness 下的 no-op 接回基线：

- HOTA = 52.758
- AssA = 44.038
- IDF1 = 58.276
- MOTA = 73.232
- IDSW = 847

### 5.2 `rerank_only`

较平滑的 local residual 重排：

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

### 5.3 `rerank_minimal`

更保守的 operator：
- hard trigger
- minimal winner override
- margin < 0.10

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

### 5.4 `oracle_rerank`

这是最高信息增益的诊断实验：
- top-k 对齐到 8
- primary-only
- pre-Hungarian
- rerank-only
- minimal winner override
- winner 直接用 replay label oracle

结果：
- HOTA = 52.220
- AssA = 43.202
- IDF1 = 58.316
- MOTA = 72.753
- IDSW = 1012

相对 `noop`：
- HOTA -0.538
- AssA -0.836
- IDF1 +0.040
- MOTA -0.479
- IDSW +165

## 6. 我当前的结论

我现在认为，最硬的负信号已经不是 learned deployment 没调好，而是：

连 `top-8 + oracle winner + minimal winner override` 都没有打过 `noop`

这说明问题已经不只是：
- 模型没学到
- trigger 不够好
- score operator 不够平滑

而更像是：
- `single-row / row-local rerank` 这个 decision unit 本身就不值得继续当主线

## 7. 我想让你只回答最终决策问题

请你直接回答下面这些问题：

1. 基于现在这组完整证据，你是否同意：
   - 当前 `row-local learned online controller` 主线已经可以判定为 `KILL`

2. 如果你同意 kill，这个 kill 具体应当停止到什么粒度？
   请明确说清楚：
   - 是只停当前 learned deployment
   - 还是连 `single-row local rerank` 作为 decision unit 也一并停掉

3. 如果这条线应当停止，下一主线在下面几个候选里你建议哪一个？
   - `competition-aware local conflict graph`
   - `tracklet-level continuity / stitching`
   - 其他更大的 structured association decision unit

4. 请给出唯一主方向，不要列很多都可以。

5. 如果你的建议是 `competition-aware local conflict graph`，请你明确回答：
   - 为什么它比 `tracklet-level continuity` 更合理
   - 为什么它比继续 row-local learned rerank 更值得

6. 如果你不同意 kill，请你必须明确说明：
   - 在 oracle 都不打过 noop 的情况下，你仍建议继续的唯一技术理由是什么

## 8. 随附材料

你可以结合下面这些压缩包一起判断：

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
- online oracle_rerank bundle：
  - `outputs/competition_assoc_online_bundle_20260323_142134.zip`

请不要只给“也许还可以试试”的模糊建议。我现在要的是基于 oracle negative signal 的最终 kill / next-mainline 决策。
```
