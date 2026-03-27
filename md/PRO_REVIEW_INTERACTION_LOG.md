# Pro Review Interaction Log

这份文档记录每次向 Pro 发送的提问，以及 Pro 给出的回答。它不是可有可无的备忘录，而是后续所有新 Pro 必须继承的上下文链。

## 强制规则

- 每次与 Pro 聊天后，必须追加一条记录，不能隔天补。
- 记录对象同时包括:
  - 我们发给 Pro 的问题
  - Pro 的回答
  - 我们实际采纳了什么
  - 后续证据有没有修正这次回答
- 如果某次 Pro 建议后来被实验否掉，也不要删除，直接在该条下补 `后续证据修正`。
- 新的 Pro 提问时，默认把这份日志一并带上。

## 每条记录至少应包含

- 日期
- 提问主题
- 对应 prompt 文件
- 提问摘要
- Pro 回答摘要
- Pro 的核心判断
- 我们最终采纳的动作
- 后续证据修正

## 推荐做法

- 如果本次提问和回答已经单独保存成文件，也把文件路径写进记录。
- 如果这次回答直接影响代码或实验队列，记录里要写清楚最终落到了哪个脚本、配置或输出目录。
- 如果这次没有采纳 Pro 的建议，也要明确写 `未采纳`，不要留空。

## 2026-03-23

### 主题

local conflict graph 是否应成为新主线，以及应该如何重设计。

### 对应提示词

- `md/PRO_REVIEW_SEND_TO_PRO_LOCAL_CONFLICT_GRAPH_REDESIGN_20260324.md`

### 提问摘要

- 当前 row-local controller 已经拿到负证据。
- 需要判断 local conflict graph 是否应成为新主线。
- 需要 Pro 直接输出代码级重设计文档。

### Pro 核心回答

- 判定: `NARROW GO`
- row-local rerank 的 decision unit 已错，不应再继续磨。
- 应把主 decision unit 从 single-row 改成 cluster-level local assignment。
- 应优先做 full cluster replacement oracle，先验证 cluster 级完整决策单元本身有没有 ceiling。
- continuity / stitching 不该进当前主故事。

### 当时采纳的动作

- 采纳了“主决策单元要升级到 cluster-level”的判断。
- 先推进了 local conflict graph 相关实现与 oracle 路径。

### 后续证据修正

- full cluster replacement oracle 后续被在线负证据否掉，不能继续作为主语义。
- 但 cluster-level decision unit 本身并没有被否掉，只是部署语义需要收缩。

## 2026-03-24

### 主题

在 full replacement 负证据出现后，当前主线该如何重新收束。

### 对应提示词

- 口径已整合进当前仓库中的 redesign 相关文档与实现说明。

### 提问摘要

- full cluster replacement 已被在线负证据否掉。
- 需要判断 cluster-level 主线是否仍然成立。
- 需要明确新的在线语义和训练目标该如何重写。

### Pro 核心回答

- 判定: `NARROW GO`
- 被否掉的是“整块替换 host”，不是 cluster 决策本身。
- 新模块不该学 full replacement，而该学 conservative partial commit。
- 主模块应升级为:
  - learned local assignment
  - conservative commit operator
  - defer-to-host 语义
- 模型输出不应再是 keep / rerank / null，而应是 `[local tracks + defer]` 上的 assignment。

### 我们最终采纳的动作

- 正式把主线改成 `LocalConflictCommitRefiner`。
- 在线语义固定为 `partial commit + defer to host`。
- 在代码中加入 `learned_commit` mode。
- 补了 dataset builder、trainer、proxy runner、generic runner、12h queue。

## 2026-03-24 后续实验反馈

### 当前实验事实

- learned commit 已在 proxy0213 上超过 oracle hard-trigger control。
- full FRCNN 已至少在 `md2/mm2` 上拿到小幅正号。

### 当前解释

- conservative partial commit 这个 operator 有真实价值。
- 现在还不能说论文已经成型，但已经不只是“方向证明”。

## 2026-03-24

### 主题

learned commit 队列结束后，下一步应优先做 host migration、扩数据还是强模型。

### 对应提示词

- `md/PRO_REVIEW_SEND_TO_PRO_AFTER_LEARNED_COMMIT_QUEUE_20260324.md`

### 提问摘要

- 当前 learned commit 已在 base_reid_da 上拿到真实但不大的正号。
- 需要只在 `A. stronger host migration / B. larger training data / C. model strengthening` 中选一个主方案。
- 如果选择 A，需要在 `v13 / v15 / v16` 中选一个主选和一个备选。

### Pro 回答摘要

- 判定仍是 `NARROW GO`。
- 当前唯一主方案应是 `A. stronger host migration`。
- A 里主选 `v15_laplace_reid_da_val0213`，备选 `v13_tf_only_val0213_reid_da`，不选 `v16_laplace_trainable_val0213`。
- 下一步唯一 first-priority experiment 是 `v15` 上的 zero-shot paired proxy0213 migration。

### Pro 核心回答

- 当前 learned commit 已经回答了“在 base_reid_da 上有没有 operator 价值”，答案是有，但不大。
- 下一步最高信息增益不在当前 host 上继续堆模型或堆小 sweep，而是先回答可迁移性。
- v15 是最干净的 stronger-host 最小迁移；v16 会把 host retraining 和 operator migration 混在一起。

### 我们最终采纳的动作

- 采纳 `A. stronger host migration` 为下一步主动作。
- 先实现 `v15` 的 paired proxy0213 host-migration runner。
- 保留 `v13` 作为备选，不把 `v16` 作为第一次迁移实验。

### 后续证据修正

- 已按该建议完成 `v15` paired proxy0213 host migration。
- 结果为负，不是空跑:
  - `delta_HOTA = -0.219`
  - `delta_AssA = -0.419`
  - `delta_IDF1 = -0.121`
  - `eligible_clusters = 5769`
  - `replaced_clusters = 138`
- 这说明 `v15` 上的 first-shot zero-shot portability 没过。
- 当前新的单一决策点变成: 是否直接切 `B. larger training data`，还是还要补打一枪 `v13`。

### 相关文件

- prompt: `md/PRO_REVIEW_SEND_TO_PRO_AFTER_LEARNED_COMMIT_QUEUE_20260324.md`
- raw_answer: `md/PRO_REVIEW_REPLY_AFTER_LEARNED_COMMIT_QUEUE_20260324.md`
- run_root: `outputs/local_conflict_graph_hostmig_v15_proxy0213_20260324_194915`

## 2026-03-24

### 主题

`v15` zero-shot host migration 已负之后，下一步是直接转 `B. larger training data`，还是还要补 `v13` portability check。

### 对应提示词

- `md/PRO_REVIEW_SEND_TO_PRO_AFTER_V15_HOST_MIGRATION_NEGATIVE_20260324.md`

### 提问摘要

- `v15 host_only` 相对 `v15 + learned_commit` 的 paired delta 为负。
- diagnostics 非空，说明不是模块没触发。
- 需要在 `directly switch to B` 和 `still run v13 once` 之间选唯一主方案。

### Pro 回答摘要

- 判定仍是 `NARROW GO`。
- 近期唯一主方案是 `B. larger training data`。
- `v13` 只保留为备选，不再是当前 first priority。
- 中期唯一更强模块建议是 `LocalConflictSetPredictor`。

### Pro 核心回答

- 当前 `v15` 负迁移已经足够说明 current tiny-data / v1 checkpoint 不具备 zero-shot portability。
- 再补 `v13` 的额外信息增益，不足以改变近期工程动作。
- 近期 first-priority experiment 是 `large_data_base_retrain_v1`。
- 中期应把当前 `LocalConflictCommitRefiner` 升级成 host-conditioned 的 `LocalConflictSetPredictor`。

### 我们最终采纳的动作

- 先入档这次 Pro 回复。
- 近期按 `B. larger training data` 准备改 dataset / training pipeline。
- 保留 `v13` 为备选，不作为当前第一枪。

### 后续证据修正

- 暂无

### 相关文件

- prompt: `md/PRO_REVIEW_SEND_TO_PRO_DECISION_AND_OPEN_REDESIGN_AFTER_V15_NEGATIVE_20260324.md`
- raw_answer: `md/PRO_REVIEW_REPLY_AFTER_V15_NEGATIVE_20260324.md`

## 2026-03-25

### 主题

在 larger-data `v1` 队列进行中时，直接为下一代 stronger `v2` 模块定结构、loss、特征和 runtime 接口。

### 对应提示词

- `md/PRO_REVIEW_SEND_TO_PRO_STRONGER_V2_WHILE_LARGEDATA_RUNNING_20260324.md`

### 提问摘要

- `large_data_base_retrain_v1` 已经修正了 tiny-data / bad split。
- 需要并行确定下一代更强模块，而不是等大数据回测完全结束后再设计。
- 问题被收紧为:
  - 在 operator 语义固定不变前提下
  - 如何设计一个明显强于 `LocalConflictCommitRefiner v1` 的 `v2`
  - 并明确 loss / feature / host-shift robustness / runtime 接口 / first experiment。

### Pro 回答摘要

- 判定仍是 `NARROW GO`。
- 近期不要再补 `v13` portability check。
- 近期唯一主方案仍是 `B. larger training data`。
- 中期唯一更强模块是 `HostConditionedLocalConflictSetPredictor`。

### Pro 核心回答

- current `v1` 的主病因不是只有数据量，还包括:
  - raw host-sensitive feature semantics
  - loss 太弱
  - 结构表达不足
- `v2` 应升级为 host-conditioned local set predictor:
  - cluster-local
  - row/column-aware
  - set prediction 风格
  - 保持 `partial commit + defer-to-host`
- loss 必须从单一 row-wise CE 升级成:
  - assignment CE
  - edge auxiliary loss
  - cluster safety / gate loss
  - conservative margin loss

### 我们最终采纳的动作

- 不再补 `v13` portability。
- 直接实现完整 `set_predictor_v2` 代码链，而不是最小 MVP。
- 已落地:
  - `models/local_conflict_set_predictor.py`
  - `scripts/build_local_conflict_set_predictor_dataset_manifest.py`

### 后续证据修正

- 已完成 smoke 验证:
  - dataset builder 正常
  - trainer 正常
  - non-finite 问题已修
- 正式 large-base `v2` 队列已启动:
  - `outputs/local_conflict_set_predictor_large_base_20260325_013200`
  - 当前应以该目录下的 `summary.csv` 为准

### 相关文件

- prompt: `md/PRO_REVIEW_SEND_TO_PRO_STRONGER_V2_WHILE_LARGEDATA_RUNNING_20260324.md`
- raw_answer: `md/PRO_REVIEW_REPLY_STRONGER_V2_WHILE_LARGEDATA_RUNNING_20260325.md`

## 2026-03-25

### 主题

在 baseline hierarchy 已固定为 `official ByteTrack` 后，strict official-host retrain 结果显著为负，下一次向 Pro 需要收窄到 official ByteTrack 主线下的 redesign。

### 对应提示词

- `md/PRO_REVIEW_SEND_TO_PRO_AFTER_OFFICIAL_BYTETRACK_STRICT_NEGATIVE_20260325.md`

### 提问摘要

- `official ByteTrack` 已固定为论文主 carrier，不再继续讨论 baseline selection。
- strict paired protocol 已经打通。
- zero-shot internal-host-trained `v2` 在 official host 上轻微负且几乎不介入。
- official-host retrain 已经成功，但 paired half-val 显著为负，而且 diagnostics 显示模块大规模介入。
- 需要 Pro 不再讨论 baseline，而只回答:
  - 当前 negative 的主病因更接近 supervision / gate / runtime / family 中的哪一层
  - 下一步唯一主 redesign 应该是什么
  - 文件级落点和第一枪实验如何写

### Pro 回答摘要

- 判定为 `NARROW GO`。
- 当前主病因不是 `set_predictor_v2` family 本身，而是 official-host 下的 dataset / target semantics 已把模块训成 aggressive replacer。
- 唯一主方案是保留 `set_predictor_v2` family，但把它改造成 `Utility-Gated Delta-Commit Set Predictor`。
- 备选不是换 family，而是只在 runtime 上包一层更强的 conservative wrapper。

### Pro 核心回答

- 当前 `trigger_pass` / `oracle_commit_pairs` 语义表达的是“这个 cluster 是否存在 GT 合法完整解”，而不是“在 official ByteTrack 当前 host 上是否值得介入”。
- official-host dataset 中 `trigger_fail_clusters = 0`，gate negatives 基本缺失，直接导致 `cluster_commit_logit` 学成“eligible 即开”。
- 训练目标应从 `full oracle commit` 改成 `delta commit against host`：
  - 先构造 `host_pairs_local`
  - 再构造 `oracle_pairs_local`
  - 用 `delta_commit_pairs = oracle_pairs_local - host_pairs_local` 作为插件应提交的增量匹配
  - 其余 detection 一律监督为 `defer`
- cluster gate 目标应从 `trigger_pass` 改成 `cluster_should_intervene` / utility-style label，而不是继续学“有合法匹配就开”。
- runtime 需要叠加更强的 conservative constraints：
  - 更严格的 utility gate
  - replacement budget
  - per-cluster max commits
  - stronger abstain / margin filtering
- 下一步唯一 first-priority experiment 应是：
  - `official ByteTrack host-only` vs `utility-gated delta-commit plugin`
  - 同 detector / checkpoint / split / evaluator / pre-Hungarian injection 的 strict half-val paired eval。

### 我们最终采纳的动作

- 已将本次 Pro 回答保存为:
  - `official_bytetrack_redesign_decision_20260325.md`
- 采纳该回答作为当前唯一主 redesign 文档。
- 下一步工程动作固定为：
  - 先改 `scripts/build_local_conflict_set_predictor_dataset.py`
  - 再改 `scripts/train_local_conflict_set_predictor.py`
  - 再接入 `models/local_conflict_set_predictor.py`
  - 再改 `third_party/ByteTrack/yolox/tracker/byte_tracker_local_conflict.py`
  - 最后更新 official ByteTrack half-val paired runner。

### 后续证据修正

- 当前尚未被后续实验修正。
- 本次回答替换了先前“优先做 gate calibration / v2.1 小修”的收束方向；新的首要任务已改为 `delta-utility teacher + conservative runtime`。
- 相关文件将以本条记录为新的实现锚点持续更新。

### 相关文件

- prompt: `md/PRO_REVIEW_SEND_TO_PRO_AFTER_OFFICIAL_BYTETRACK_STRICT_NEGATIVE_20260325.md`
- raw_answer: `official_bytetrack_redesign_decision_20260325.md`
- run_root:
  - `outputs/official_bytetrack_local_conflict_halfval_pair_20260325_184000`
  - `outputs/official_bytetrack_local_conflict_stage1_trainhalf_20260325_195300`

## 2026-03-25

### 主题

stable `v2` base-host 结果已完成后，收束询问“当前有效模块的下一步唯一增强点是什么”。

### 对应提示词

- `md/PRO_REVIEW_SEND_TO_PRO_AFTER_STABLE_V2_BASE_RESULTS_20260325.md`

### 提问摘要

- `stable v2` 已经完整收尾，不再是中途状态。
- 当前关键事实:
  - stable `proxy0213`: `53.118 / 44.577 / 58.730 / 73.437 / 811`
  - stable `full md2/mm2`: `63.257 / 60.191 / 72.128 / 76.055 / 1481`
  - 相对 enlarged-data `v1`:
    - `full` 仍然为正
    - `proxy` 收缩明显
- 当前问题不再是“要不要这条线”，而是:
  - 为什么 stable `full` 还正、stable `proxy` 却收缩
  - 下一步唯一 first-priority enhancement 应该落在 `gate / loss / feature / proxy-full mismatch` 里的哪一个

### Pro 回答摘要

- 待回复

### Pro 核心回答

- 待回复

### 我们最终采纳的动作

- 已准备新的收束型 Pro 提示词。
- 不再向 Pro 提开放式“重新发明一个更强模块”问题。
- 新问题只允许 `v2.1` 级别增强，不允许重开大架构。

### 后续证据修正

- stable run root:
  - `outputs/local_conflict_set_predictor_large_base_stable_20260325_023500`
- 当前这是 base-host `v2` 的权威结果来源。

### 相关文件

- prompt: `md/PRO_REVIEW_SEND_TO_PRO_AFTER_STABLE_V2_BASE_RESULTS_20260325.md`
- raw_answer: `待补`

## 下次新增记录模板

复制下面结构追加:

```markdown
## YYYY-MM-DD

### 主题

[一句话]

### 对应提示词

- `md/...`

### 提问摘要

- ...

### Pro 回答摘要

- ...

### Pro 核心回答

- ...

### 我们最终采纳的动作

- ...

### 后续证据修正

- ...
```

## 2026-03-24

### 主题

在 `large_data_base_retrain_v1` 正在运行时，并行请求 stronger-`v2` 模块设计。

### 对应提示词

- `md/PRO_REVIEW_SEND_TO_PRO_STRONGER_V2_WHILE_LARGEDATA_RUNNING_20260324.md`

### 提问摘要

- 当前 `large_data_base_retrain_v1` 已经把 larger-data + strict sequence split 跑起来，stage1 也已完成。
- 当前新证据表明:
  - `train_examples = 817`
  - `val_examples = 286`
  - `best_epoch = 5`
  - `val_commit_precision = 0.4736`
  - `val_commit_recall = 0.8366`
- 但最终 proxy/full eval 仍在运行中。
- 需要 Pro 在“不改变 operator 语义”的前提下，直接给出 stronger-`v2` 的具体工程设计。

### Pro 回答摘要

- 待回复

### Pro 核心回答

- 待回复

### 我们最终采纳的动作

- 已准备新的 Pro bundle 与提示词。
- 当前实验队列继续运行，不因为这次并行提问而中断。

### 后续证据修正

- 待回复

## 2026-03-25

### 主题

论文 baseline family 选择与主 carrier 固定。

### 对应提示词

- `md/PRO_REVIEW_SEND_TO_PRO_BASELINE_SELECTION_AND_PAPER_PROTOCOL_20260325.md`

### 提问摘要

- 当前 `v2` 已在内部宿主线 `base_reid_da` 上验证为有效，但该宿主线不是严格论文 baseline。
- 需要在 `official ByteTrack / BoT-SORT / StrongSORT / MOTIP / 其他候选 family` 中，选择最适合作为论文 primary carrier 的 baseline。
- 需要明确 `primary baseline / secondary transfer baseline / internal ablation-only baseline / currently exclude`。

### Pro 回答摘要

- 待 Pro 回复。

### Pro 核心回答

- 待 Pro 回复。

### 我们最终采纳的动作

- 已整理开放式 baseline-selection prompt 与轻量证据包，准备发给 Pro。

### 后续证据修正

- 当前 stable `v2` 内部线有效，但论文级 baseline 仍未固定。

### 相关文件

- prompt: `md/PRO_REVIEW_SEND_TO_PRO_BASELINE_SELECTION_AND_PAPER_PROTOCOL_20260325.md`

## 2026-03-25

### 主题

official ByteTrack `delta_utility` 已从 aggressive replacer 摆到 near no-op 后，下一步 supervision / objective 应如何重写。

### 对应提示词

- `md/PRO_REVIEW_SEND_TO_PRO_AFTER_OFFICIAL_DELTA_UTILITY_NOOP_20260325.md`

### 提问摘要

- strict official ByteTrack paired protocol 已完全打通。
- old official-host retrain 是 `aggressive replacer`：
  - `replaced_clusters=5935`
  - paired delta `HOTA -3.669 / AssA -6.496 / IDF1 -4.715`
- new `delta_utility + conservative runtime` 又变成 `near no-op`：
  - `eligible_clusters=7015`
  - `gate_pass_clusters=3245`
  - `replaced_clusters=0`
  - `matched_dets=0`
  - stage1 `val_commit_precision / recall / cluster_gate_utility_cal = 0`
- 需要 Pro 直接判断：
  - 是不是 family 该换
  - 还是 supervision / objective geometry 该重写
  - 下一步 first-priority redesign 应该落到哪个具体 teacher / loss / runtime 组合

### Pro 回答摘要

- 判定：`NARROW GO`
- 不是 baseline 问题，不是 integration 问题，也不是 current family 先验被 kill。
- `hard delta = oracle - host` 的主要问题不只是过稀，而是它只表达 added pairs，不表达 host row action 的 edit utility。
- 下一步唯一主方案应保留 `HostConditionedLocalConflictSetPredictor` family，不换 backbone，把 teacher 改成 `edit-aware weighted utility target`，并把 assignment 改成 masked decision-row CE。

### Pro 核心回答

- 新方案名：
  - 概念名：`Edit-Utility Gated Selective Commit Predictor`
  - 代码建议名：`set_predictor_v2_editutility`
- dataset builder 要新增：
  - `host_action_by_det`
  - `oracle_action_by_det`
  - `row_edit_mask`
  - `row_action_type = keep | add_commit | reassign_commit | force_defer`
  - `edit_commit_pairs`
  - `cluster_edit_gain / cost / utility_gain`
  - `cluster_should_intervene`
- trainer 要新增：
  - masked decision-row CE
  - edit-aware edge target
  - target coverage band
  - coverage-aware checkpoint selection
- runtime contract 继续保留 `pre-Hungarian / partial commit / defer to host`，只补更细的 no-op diagnostics，不再把 runtime wrapper 当第一杠杆。

### 我们最终采纳的动作

- 已把 Pro 原始回复归档到：
  - `md/PRO_REVIEW_REPLY_AFTER_OFFICIAL_DELTA_UTILITY_NOOP_EDITUTILITY_20260325.md`
- 采纳 `edit_utility` 为 next mainline redesign。
- 直接在以下路径落地：
  - `scripts/build_local_conflict_set_predictor_dataset.py`
  - `scripts/train_local_conflict_set_predictor.py`
  - `third_party/ByteTrack/yolox/tracker/byte_tracker_local_conflict.py`
  - `scripts/run_official_bytetrack_local_conflict_stage1_trainhalf.py`
  - `scripts/run_official_bytetrack_local_conflict_halfval_pair.py`

### 后续证据修正

- 待本轮 `official ByteTrack host-only` vs `set_predictor_v2_editutility` strict half-val paired experiment 跑完后补。

### 相关文件

- prompt: `md/PRO_REVIEW_SEND_TO_PRO_AFTER_OFFICIAL_DELTA_UTILITY_NOOP_20260325.md`
- raw_answer: `md/PRO_REVIEW_REPLY_AFTER_OFFICIAL_DELTA_UTILITY_NOOP_EDITUTILITY_20260325.md`
