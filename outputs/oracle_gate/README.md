# Oracle Gate

这是 SPOT oracle 收口的正式汇总目录。

## 目录内容

- `summary.csv`: 当前阶段状态总表。
- `decision.md`: 当前可执行结论。

## 状态说明（2026-06-26 更新）

- `protocol_lock=completed`
- `gt_alignment=completed_verified`
- `oracle_0A=completed_positive`（7.29% IDSW reduction）
- `oracle_0C_inline_gt=completed_partial_trusted`（43.28% fixable）
- `oracle_0E=closed`（`SPOT_MAINLINE`, `runtime_patch_allowed=1`）
- `final_decision=SPOT_MAINLINE`

## 当前决策

Oracle Gate 已闭合。可以进入 P4 ADG-freeze / State Protection 最小 runtime 实现。

## 读取顺序

1. `summary.csv`
2. `decision.md`
3. `docs/oracle_gate_recap_2026-06-25.md`
4. `docs/current_mainline_2026-06-26.md`
