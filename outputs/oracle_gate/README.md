# Oracle Gate

这是 SPOT oracle 收口的正式汇总目录。

## 目录内容

- `summary.csv`: 当前阶段状态总表。
- `decision.md`: 当前可执行结论。

## 状态说明（2026-06-26 修正）

- `protocol_lock=completed`
- `gt_alignment=completed_verified`
- `oracle_0A=completed_positive`（7.29% oracle ceiling，**不是运行时增益**）
- `oracle_0C_inline_gt=completed_partial_trusted`（43.28% fixable）
- `oracle_0E=provisional`（`SPOT_PROVISIONAL`, `runtime_patch_allowed=0`）
- `final_decision=SPOT_PROVISIONAL`

## ⚠️ 重要修正

Oracle ceiling ≠ 运行时增益。0A 的 7.29% 是上界，不是真实 IDSW 降幅。
运行时补丁需要真实成对评测才能解锁。

## 当前决策

Oracle Gate 处于 PROVISIONAL 状态。可以实现最小 P4 ADG-freeze + 跑 paired eval，但不能解锁 runtime patch 直到 paired eval 正向。

## 读取顺序

1. `summary.csv`
2. `decision.md`
3. `docs/oracle_gate_recap_2026-06-25.md`
4. `docs/current_mainline_2026-06-26.md`
