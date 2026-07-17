# FM-Track / MOT20 M23-1 Notion Sync Pending

日期：2026-07-17

目标页面：`39dabff3-387a-8125-99da-caec2ec8f7ec`

状态：待同步。当前会话没有可写 Notion connector，因此未声称已写入。

## 标题

M23-1：MOT20 Pre-NMS Suppressed-Evidence Oracle Audit

## 核心结果

- 真正 pre-NMS candidates：10,334,053。
- Post-NMS kept：1,188,001。
- NMS suppressed：9,146,052。
- 8,931 帧均与冻结 Phase-0 严格等价。
- GT-free fixed-budget candidates：459,899。
- Candidate pool ratio：1.499998637，预算通过。
- Post-NMS replace/add oracle：83.676 HOTA。
- Pre-NMS budget replace/add oracle：84.523 HOTA。
- Pre-NMS budget selected ceiling：85.132 HOTA。
- Full pre-NMS selected ceiling：85.915 HOTA。
- 最差序列：MOT20-03，budget selected ceiling 84.641。
- Combined 84.5 与 worst-sequence 84.0 均通过。
- Deployment：false。

## 关键解释

Pre-NMS budget 将 FN 从 30,071 降到 18,630，而 FP 仅从 17,753 增到 17,755。新增恢复的 17,715 个观测中，57.26% score >=0.60，63.25% 与 suppressor IoU >=0.90。说明拥挤场景中的 NMS 会删除高置信、身份相关或定位更优的候选，不能把它们简单视为重复框。

## 决策

`expanded_evidence_ceiling_sufficient = true`。

下一阶段进入 M23-2：在固定 459,899 个 budget candidates 上进行 appearance-conditioned observation selection，并把被授权观测接入 global tracklet graph。不得再扩大候选池，不做 NMS/score sweep。

## 复现

- Candidate plan 43/43 文件字节一致。
- Oracle compact 7/7 文件字节一致。
- 28/28 tracker hashes 一致。
- 7/7 TrackEval summaries 一致。

## 路径

- `outputs/mot20_m23_20260717/prenms_suppression_v1`
- `outputs/mot20_m23_20260717/prenms_candidate_plan_v1`
- `outputs/mot20_m23_20260717/prenms_suppressed_oracle_v1`
- `deliverables/mot20_m23_prenms_suppressed_evidence_oracle_20260717.md`

## Lock state

- P15 policy：no-op。
- Locked-label reads：0。
- Locked TrackEval calls：0。
- Remaining locked rows untouched：156。
