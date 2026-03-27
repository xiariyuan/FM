# Runtime Replay Diagnostic Findings 2026-03-19

## 1. 诊断对象

- 宿主: `external/sw_yolox`
- 插件: `runtime_replay_hardtrain_mixedval_gate.pt`
- 重点序列:
  - 正样本: `MOT17-02-FRCNN`, `MOT17-13-FRCNN`
  - 回撤/难样本: `MOT17-09-FRCNN`, `MOT17-11-FRCNN`

## 2. Runtime dump 分析结论

来自 `outputs/runtime_replay_diag_dump_20260319/runtime_dump_analysis.json`:

- `MOT17-02-FRCNN`
  - flip_rate `0.01068`
  - ambiguous_rate_005 `0.14349`
  - mean_base_margin `0.20849`
- `MOT17-13-FRCNN`
  - flip_rate `0.00430`
  - ambiguous_rate_005 `0.07692`
  - mean_base_margin `0.23564`
- `MOT17-09-FRCNN`
  - flip_rate `0.00461`
  - ambiguous_rate_005 `0.03447`
  - mean_base_margin `0.54186`
- `MOT17-11-FRCNN`
  - flip_rate `0.00173`
  - ambiguous_rate_005 `0.01565`
  - mean_base_margin `0.61643`

## 3. 直接解释

- `02/13` 的歧义组比例明显更高，runtime replay 确实有更多可发挥空间。
- `09/11` 大多数 group 本来就很容易，base margin 很大，真正需要 rerank 的 group 很少。
- 因此 full7 上的主要问题不是“插件在 easy groups 上过度触发很多次”，而是:
  - 插件只在很少数歧义组里有机会发挥
  - 这些少数歧义组上的判别质量还不够强，无法在完整 full7 上形成更大的总增益

## 4. Hard gate 实验

我加了一个 runtime hard ambiguity gate:

- 新开关:
  - `ASSOC_RUNTIME_REPLAY_HARD_MARGIN_GATE`
  - `ASSOC_RUNTIME_REPLAY_MARGIN_THRESHOLD`
- 实现:
  - 在 runtime 中，如果当前 detection group 的 base top1-top2 margin 大于阈值，则直接 no-op

### 结果

子集 `02/09/11/13` 上:

- 原 replay:
  - HOTA `70.670`
  - AssA `64.623`
  - IDF1 `79.605`
  - IDSW `221`
- hard-gate replay:
  - HOTA `70.670`
  - AssA `64.623`
  - IDF1 `79.605`
  - IDSW `221`

逐序列也完全一致。

## 5. 结论

hard gate 完全无效，说明:

- 当前模型本身已经基本只在少量歧义组上起作用
- full7 的小回撤不是靠“再保守一点”能解决的

真正下一步应该是:

- 改歧义组内部的训练对象和判别质量
- 不是继续堆更强的 easy-group 抑制

## 6. 推荐的下一步

- 用 runtime dump 直接重构训练集:
  - 重点抽取 `base_margin < 0.05` 的 hard ambiguous groups
  - 对 `09/11` 这类回撤序列提高采样权重
- 训练目标从“平均 proxy gain”改成:
  - hard-group top1 correction
  - selected-vs-rival margin gain
  - easy-group strict no-op 只作为辅损失，不再作为主手段
- 新实验优先级:
  - `runtime hard-negative replay retrain`
  - 再跑 `02/09/11/13`
  - 再跑 `full7`
