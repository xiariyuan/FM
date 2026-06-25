# SPOT Oracle Gate Recap

**Date:** 2026-06-25  
**Scope:** `MOT20-05` oracle evidence收口，按 protocol lock 只整理证据，不扩展 runtime tracker patch。

## 结论先行

当前这轮 oracle 工作是**部分完成，但未正式收口**。

已确认的事实：

- `protocol lock` 已落盘并生效。
- `GT alignment` 已完成，且结果稳定。
- `Oracle 0A` 在真实 `MOT20-05` 上给出了正信号，可继续作为证据链的一部分。
- `Oracle 0C` 的 full-file 运行已被手动中止，结构化记录应保持 `interrupted`。
- `Oracle 0C` 的 chunked partial 只证明流程跑通，**不能当最终结论**。
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

### Oracle 0C

- `outputs/spot_oracle_0C_mot20_05_pairbank_20260624/summary.csv`
- `outputs/spot_oracle_0C_mot20_05_f0120_partial_20260624/oracle_cost_rerank_report.md`
- `outputs/spot_oracle_0C_mot20_05_f0120_partial_20260624/summary.csv`

### Oracle 0B / 0D / 0E smoke

- `outputs/spot_oracle_0B_smoke_20260624/oracle_delay_report.md`
- `outputs/spot_oracle_0D_smoke_20260624/oracle_false_positive_freeze_report.md`
- `outputs/spot_oracle_0E_smoke_20260624/joint_oracle_decision.md`

## 当前状态判断

### 已完成

1. 协议锁定。
2. GT 对齐。
3. Oracle 0A 正信号采集。
4. 0B / 0D / 0E smoke 链路跑通。

### 未收口

1. `Oracle 0C` full-file 主结论缺失。
2. `Oracle 0C` partial 的 `fixable_percent=99.982919` 明显偏乐观，不能直接作为决策依据。
3. 当前没有一个正式的 `outputs/oracle_gate/` 总目录来承接最终决策。

### 不能误写的状态

- `0C full-file` 不能写成 `completed`。
- `0C partial` 不能写成 `final`.
- `smoke` 不能写成 `real MOT20-05` 结论。

## 推荐后续动作

1. 重新做一个可信的 `0C` 证据链，或者明确说明为什么 `0C` 不能用于决策。
2. 把 `0A / 0B / 0C / 0D / 0E` 汇总到正式的 `outputs/oracle_gate/` 目录。
3. 只有在证据链闭合后，再考虑解除 runtime patch 冻结。

## 上传边界

这次已上传的是代码、协议文档和小型结构化结果记录。下列大文件不纳入 GitHub 主仓库：

- `outputs/spot_alignment_mot20_05_baseline_20260624/reports/gt_alignment.json`
- `outputs/spot_alignment_mot20_05_baseline_20260624/reports/gt_alignment_rows.csv`
- `outputs/tos_analysis_v5_20260618_084245/tos_analysis/MOT20-05_frames.csv`
- `outputs/spot_oracle_0C_mot20_05_f0120_partial_20260624/fixable_events.json`

这些文件对继续工作不是必须的，保留在本地即可。
