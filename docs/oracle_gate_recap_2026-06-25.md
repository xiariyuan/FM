# SPOT Oracle Gate Recap

**Date:** 2026-06-25 (corrected 2026-06-26)  
**Scope:** `MOT20-05` oracle evidence收口，按 protocol lock 只整理证据，不扩展 runtime tracker patch。

## ⚠️ CORRECTION (2026-06-26)

Previous version incorrectly stated `runtime_patch_allowed=1` and `SPOT_MAINLINE`.
Oracle ceiling ≠ runtime gain. Runtime patches require real paired eval to unlock.

## 结论先行

当前这轮 oracle 工作是**已完成，但处于 PROVISIONAL 状态**。

已确认的事实：

- `protocol lock` 已落盘并生效。
- `GT alignment` 已完成，且结果稳定。
- `Oracle 0A` 在真实 `MOT20-05` 上给出了 oracle ceiling (7.29% recoverable rate)，**这是上界，不是运行时增益**。
- `Oracle 0C` 的 full-file 运行已被手动中止，结构化记录应保持 `interrupted`。
- `Oracle 0C` 的 inline GT 版本已完成，`fixable_percent=43.28%`，这是一个中等强度的信号。
- `Oracle 0E` 的最终决策是 `SPOT_PROVISIONAL`，`runtime_patch_allowed=0`。
- `Oracle 0B / 0D / 0E` 目前是 smoke 级或决策级辅助证据，不应被写成真实主线结论。

## 关键文件

### Protocol

- `outputs/spot_protocol_smoke_20260624/protocol_lock.md`
- `outputs/spot_protocol_smoke_20260624/summary.csv`

### GT Alignment

- `outputs/spot_alignment_mot20_05_baseline_20260624/reports/gt_alignment_report.md`
- `outputs/spot_alignment_mot20_05_baseline_20260624/summary.csv`

### Oracle 0A

- `outputs/spot_oracle_0A_mot20_05_baseline_20260624/oracle_state_protection_report.md`
- `outputs/spot_oracle_0A_mot20_05_baseline_20260624/summary.csv`

### Oracle 0C (inline GT)

- `outputs/spot_oracle_0C_mot20_05_inline_gt_20260625/oracle_cost_rerank_report.md`
- `outputs/spot_oracle_0C_mot20_05_inline_gt_20260625/oracle_cost_rerank_metrics.json`
- `outputs/spot_oracle_0C_mot20_05_inline_gt_20260625/summary.csv`

### Oracle 0E

- `outputs/spot_oracle_0E_mot20_05_allow_partial_20260625/joint_oracle_decision.md`
- `outputs/spot_oracle_0E_mot20_05_allow_partial_20260625/summary.csv`

### Oracle Gate Decision

- `outputs/oracle_gate/decision.md`
- `outputs/oracle_gate/summary.csv`

## 当前状态判断

### 已完成

1. 协议锁定。
2. GT 对齐。
3. Oracle 0A oracle ceiling 采集。
4. Oracle 0C inline GT 证据采集。
5. 0B / 0D / 0E smoke 链路跑通。
6. Oracle Gate 决策文件已修正。

### 未收口

1. `Oracle 0E` 仍然 `provisional`，`runtime_patch_allowed=0`。
2. 需要真实 paired eval 才能解锁 runtime patch。

### 不能误写的状态

- `0C full-file` 不能写成 `completed`。
- `0C inline GT` 是 partial，不能写成 `final`.
- `smoke` 不能写成 `real MOT20-05` 结论。
- `oracle_recoverable_rate` 是上界，不能写成 `IDSW reduction`。

## 推荐后续动作

1. 实现最小 P4 ADG-freeze runtime patch (appearance/history freeze only)。
2. 跑 smoke: spot_enable=0 必须与 baseline 完全一致。
3. 跑 MOT20-05 paired eval (baseline vs SPOT)。
4. 只有 paired eval 正向才解锁 runtime_patch_allowed=1。
5. 之后才扩展到 DanceTrack/SportsMOT。

## 上传边界

这次已上传的是代码、协议文档和小型结构化结果记录。下列大文件不纳入 GitHub 主仓库：

- `outputs/spot_alignment_mot20_05_baseline_20260624/reports/gt_alignment.json`
- `outputs/spot_alignment_mot20_05_baseline_20260624/reports/gt_alignment_rows.csv`
- `outputs/tos_analysis_v5_20260618_084245/tos_analysis/MOT20-05_frames.csv`
- `outputs/spot_oracle_0C_mot20_05_f0120_partial_20260624/fixable_events.json`

这些文件对继续工作不是必须的，保留在本地即可。
