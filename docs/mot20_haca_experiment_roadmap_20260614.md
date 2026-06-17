# MOT20 HACA Experiment Roadmap (2026-06-14)

## 目的

这份文档把 `ai回复1.txt` 中几家模型的分析、仓库当前真实实验记录、以及我对后续主线的判断合并到一个可执行路线里。

目标不是再讲一轮泛泛论文故事，而是明确：

1. 现在什么已经被证实。
2. 接下来先做哪些实验最值。
3. 每个实验要改什么、跑什么、看什么记录。
4. 哪些实验应该立刻停，哪些值得继续扩。

## 当前事实基线

以下结论已经被现有结构化记录直接支持。

### 1. background gate 是 MOT20 崩坏主因

证据：

- 默认带 background gate 的 MOT20 baseline 在 [outputs/sca_lmf_mot20_eval_20260609_081744/summary.csv](/gemini/code/FMtrack-main/FM-Track/outputs/sca_lmf_mot20_eval_20260609_081744/summary.csv:1) 中只有：
  - `HOTA=9.3648`
  - `AssA=2.5205`
  - `IDSW=61558`
- 同一条 HACA v3 checkpoint，在 [outputs/haca_mot20_bg_gate_probe_20260612_005759/summary.csv](/gemini/code/FMtrack-main/FM-Track/outputs/haca_mot20_bg_gate_probe_20260612_005759/summary.csv:1) 的 no-background 单序列 probe 上恢复到：
  - `HOTA=67.083`
  - `IDSW=312`
- 同一 checkpoint 在 [outputs/haca_mot20_v3_nobg_full_eval_20260612_012822/summary.csv](/gemini/code/FMtrack-main/FM-Track/outputs/haca_mot20_v3_nobg_full_eval_20260612_012822/summary.csv:1) 的全序列 no-background 评估上达到：
  - `HOTA=77.394`
  - `AssA=74.306`
  - `IDF1=88.866`
  - `IDSW=819`

判断：

- 问题不在 checkpoint 整体失效。
- 问题集中在 runtime 的 `background gate` 路径，尤其是 `group_context -> bg_prob -> (1-bg) * s_prebg` 这一支。

相关代码：

- [external/BoT-SORT-main/tracker/haca_assoc.py](/gemini/code/FMtrack-main/FM-Track/external/BoT-SORT-main/tracker/haca_assoc.py:539)
- [external/BoT-SORT-main/tracker/bot_sort.py](/gemini/code/FMtrack-main/FM-Track/external/BoT-SORT-main/tracker/bot_sort.py:1238)

### 2. no-background HACA 已经是一条健康主线

证据：

- [outputs/haca_mot20_v3_nobg_full_eval_20260612_012822/summary.csv](/gemini/code/FMtrack-main/FM-Track/outputs/haca_mot20_v3_nobg_full_eval_20260612_012822/summary.csv:1)

判断：

- 当前最应该保护的不是旧的 bg baseline，而是 `HACA v3 + competition head + no background gate` 这条线。
- 后续所有 MOT20 诊断都应该先围绕这条健康主线展开，而不是继续把时间花在已经明确崩坏的默认 bg 配置上。

### 3. Stage1 / freeze 在 MOT20 no-background 主线上目前没有收益

证据：

- [outputs/sca_lmf_mot20_nobg_eval_20260612_081934/summary.csv](/gemini/code/FMtrack-main/FM-Track/outputs/sca_lmf_mot20_nobg_eval_20260612_081934/summary.csv:1)

结果：

| variant | HOTA | AssA | IDF1 | IDSW | 相对 baseline |
|---|---:|---:|---:|---:|---:|
| baseline | 77.394 | 74.306 | 88.866 | 819 | - |
| freeze_only | 77.377 | 74.281 | 88.844 | 830 | -0.017 |
| stage1_only | 77.351 | 74.240 | 88.807 | 845 | -0.043 |
| stage1_freeze | 77.363 | 74.266 | 88.790 | 852 | -0.031 |

判断：

- 现在没有证据支持把 Stage1 或 freeze 继续当作 MOT20 主增益点。
- 它们更像是“待审查的附加模块”，而不是当前主线。

### 4. pseudotrack shard 构建仍然是高负载瓶颈

已知脚本入口：

- [scripts/run_haca_v3_mot20_train.sh](/gemini/code/FMtrack-main/FM-Track/scripts/run_haca_v3_mot20_train.sh:1)
- [scripts/build_gt_pseudotrack_groups.py](/gemini/code/FMtrack-main/FM-Track/scripts/build_gt_pseudotrack_groups.py:118)

当前默认参数：

- `--batch-size 4`
- `--max-history 8`
- `--candidate-topk 16`
- `--max-hard-negatives 6`
- `--max-random-negatives 2`

判断：

- MOT20 的压力点主要不是训练 head 本身，而是 shard 构建里的 FastReID 特征提取、候选枚举和 NPZ/CSV 物化。
- 这条线值得做受控小消融，因为它决定后续所有重训实验的时间成本和稳定性。

### 5. 记录系统存在明确 bug 和语义缺口

证据：

- [scripts/append_experiment_record.py](/gemini/code/FMtrack-main/FM-Track/scripts/append_experiment_record.py:23) 的 `--kind` 只接受 `train/eval/analysis/other`
- [scripts/run_nobg_mot20_followup.sh](/gemini/code/FMtrack-main/FM-Track/scripts/run_nobg_mot20_followup.sh:131) 却以 `--kind queue` 调用 registry 追加
- [scripts/upsert_experiment_plan.py](/gemini/code/FMtrack-main/FM-Track/scripts/upsert_experiment_plan.py:28) 只支持 `queued/running/completed/failed/cancelled`
- [scripts/run_haca_v3_mot20_train.sh](/gemini/code/FMtrack-main/FM-Track/scripts/run_haca_v3_mot20_train.sh:79) 的 `on_exit()` 将所有非零退出统一写成 `failed`

判断：

- 这不是小瑕疵，而是会直接污染后续实验判断的基础设施问题。
- 在继续大批量实验前，至少要先修一轮最小闭环。

## 总体策略

我建议按下面这个顺序推进：

1. 先修记录系统，避免继续积累错误状态。
2. 再把当前被打断的 HACA 重训链路续跑清楚。
3. 再做 pseudotrack 负载消融，降低后续所有训练成本。
4. 再做 background gate 诊断和轻量修复。
5. 再做 HACA no-background 版本矩阵，确认真正有价值的 carrier。
6. 最后才重新审查 Stage1 / freeze 是否还有 headroom。

简化地说：

- 先稳基础设施。
- 再稳 no-bg HACA 主线。
- 再审 Stage1。

## 优先级实验清单

下面每个实验都给出目的、涉及文件、建议执行方式、记录要求、成功标准和继续条件。

---

## A0. 记录系统修复 Smoke Test

### 目的

修复最小但会误导决策的记录问题，让后续每次实验都能在 `summary.csv + experiment_plan.csv + experiment_registry.csv` 里对得上。

### 涉及文件

- [scripts/append_experiment_record.py](/gemini/code/FMtrack-main/FM-Track/scripts/append_experiment_record.py:1)
- [scripts/upsert_experiment_plan.py](/gemini/code/FMtrack-main/FM-Track/scripts/upsert_experiment_plan.py:1)
- [scripts/run_nobg_mot20_followup.sh](/gemini/code/FMtrack-main/FM-Track/scripts/run_nobg_mot20_followup.sh:1)
- [scripts/run_haca_v3_mot20_train.sh](/gemini/code/FMtrack-main/FM-Track/scripts/run_haca_v3_mot20_train.sh:1)

### 要做什么

1. 统一 queue 级记录契约：
   - 方案一：给 `append_experiment_record.py` 增加 `queue`
   - 方案二：把 queue 脚本改为 `kind=other`
2. 给训练脚本增加 `interrupted` 语义，至少在 `summary.csv` 或 `notes/extra` 里能区分：
   - 外部终止
   - OOM / Killed
   - 真正代码失败
3. 回填已有错误记录：
   - `outputs/mot20_nobg_followup_20260612_081934/summary.csv`
4. 做一次最小 smoke test，验证 plan / summary / registry 三处状态一致。

### 建议执行

- 先创建一个临时输出目录，例如 `outputs/_record_smoke_<ts>/`
- 用最小 Python 调用或轻量 shell wrapper 模拟：
  - running
  - completed
  - failed

### 结构化记录要求

- 临时 smoke run 自己也要有 `summary.csv`
- 必须同时写入：
  - `outputs/experiment_plan.csv`
  - `outputs/experiment_registry.csv`

### 成功标准

1. queue 类型记录不再报错。
2. 手动重跑后，结构化记录能从 stale `failed` 改回 `running`。
3. 同一 run 不再因相对/绝对路径混用生成重复 registry 行。

### 继续条件

- 只有 A0 通过，后面的训练队列才值得继续。

---

## A1. 精确续跑当前被打断的 HACA 重训

### 目的

把当前被打断的 `outputs/haca_mot20_train_20260613_082005` 重新接回主线，确认 shard 复用、checkpoint 产物和状态记录都正常。

### 当前状态

- [outputs/haca_mot20_train_20260613_082005/summary.csv](/gemini/code/FMtrack-main/FM-Track/outputs/haca_mot20_train_20260613_082005/summary.csv:1)
  - `gt_pseudotrack=success`
  - `haca_v1=running`
  - `haca_v2=pending`
  - `haca_v3=pending`

这表明它上次中断时，至少 summary 没有被正确回填到非 running 终态。

### 涉及文件

- [scripts/run_haca_v3_mot20_train.sh](/gemini/code/FMtrack-main/FM-Track/scripts/run_haca_v3_mot20_train.sh:1)

### 要做什么

1. 在训练前增加 checkpoint 存在性检查：
   - `haca_v1/mot20_haca_v1.npz`
   - `haca_v2/mot20_haca_v2.npz`
   - `haca_v3/mot20_haca_v3.npz`
2. 如果 checkpoint 已存在且可加载，则直接将对应 phase 记为 `success` 并跳过。
3. 如果只有 shard 存在，则只跳过 shard 构建，从尚未完成的训练 phase 继续。
4. 在重跑前先把 `summary.csv` 与 `experiment_plan.csv` 回填成真实状态。

### 建议执行

- 用同一个 `OUT_ROOT=outputs/haca_mot20_train_20260613_082005`
- 先 dry read：
  - shard 是否齐全
  - `haca_v1` checkpoint 是否存在
  - `train.log` 最后一次终止位置
- 再正式重跑

### 结构化记录要求

- 继续沿用原 run 的 `summary.csv`
- 立即更新 `outputs/experiment_plan.csv`
- 完成或失败后追加/更新 `outputs/experiment_registry.csv`

### 成功标准

1. 旧 shard 被复用，不重复全量构建。
2. 最终产出新的 `mot20_haca_v3.npz`，并且状态记录一致。
3. 之后能直接接 `run_nobg_mot20_eval.sh`。

### 继续条件

- A1 完成后，no-background 主线 checkpoint 进入“可重复使用”状态。

---

## A2. pseudotrack 负载消融

### 目的

降低 MOT20 shard 构建的计算、显存和 I/O 压力，为后续多次 HACA 重训创造更轻的默认配置。

### 涉及文件

- [scripts/run_haca_v3_mot20_train.sh](/gemini/code/FMtrack-main/FM-Track/scripts/run_haca_v3_mot20_train.sh:20)
- [scripts/build_gt_pseudotrack_groups.py](/gemini/code/FMtrack-main/FM-Track/scripts/build_gt_pseudotrack_groups.py:118)

### 优先测试参数

建议先只改 3 个最有可能影响性价比的参数：

1. `candidate_topk`: `16 -> 8`
2. `max_hard_negatives`: `6 -> 3` 或 `4`
3. `batch_size`: `4 -> 8` 或 `16`，视显存而定

可选第二轮：

4. `max_history`: `8 -> 4`
5. `frame_window`: `120 -> 180` 或 `240`

### 建议实验矩阵

先做最小 4 组：

| exp_id | topk | hard_neg | batch | history | 目的 |
|---|---:|---:|---:|---:|---|
| A2a | 16 | 6 | 4 | 8 | 当前基线 |
| A2b | 8 | 6 | 4 | 8 | 看 topk 单独影响 |
| A2c | 8 | 3 | 4 | 8 | 看 pair 枚举削减 |
| A2d | 8 | 3 | 8/16 | 8 | 看吞吐提升 |

### 执行方式

- 不要一上来跑全序列全链路。
- 先选：
  - `MOT20-01 train_half f0001_0120`
  - `MOT20-05 val_half f0001_0120`
- 对每组记录：
  - wall time
  - peak GPU memory
  - 输出 NPZ 大小
  - groups / candidates 数

### 结构化记录要求

每组都产出一个小型 `summary.csv`，字段至少包括：

- `exp_id`
- `seq`
- `frame_range`
- `candidate_topk`
- `max_hard_negatives`
- `batch_size`
- `max_history`
- `wall_seconds`
- `npz_size_mb`
- `groups`
- `candidates`
- `status`

建议输出到：

- `outputs/mot20_pseudotrack_ablation_<ts>/summary.csv`

### 成功标准

1. 在不明显损伤数据覆盖的前提下，`wall time` 明显下降。
2. `candidates` 和 `npz_size` 明显下降。
3. 后续用该配置训练出来的 HACA no-bg 指标不明显退化，目标控制在 `HOTA drop <= 0.2`。

### 继续条件

- 如果 A2c/A2d 明显更轻，就把它们作为后续训练默认参数。

---

## B1. background gate 分布诊断

### 目的

把“bg gate 为什么会崩”从定性判断推进到可观测分布，确认是否真的是高 bg_prob / 大 group / 小 margin 组合导致全局抑制。

### 涉及文件

- [external/BoT-SORT-main/tracker/haca_assoc.py](/gemini/code/FMtrack-main/FM-Track/external/BoT-SORT-main/tracker/haca_assoc.py:455)
- [external/BoT-SORT-main/tracker/bot_sort.py](/gemini/code/FMtrack-main/FM-Track/external/BoT-SORT-main/tracker/bot_sort.py:1237)

### 要做什么

在 `haca_assoc.py` 的 debug 输出基础上，加一层 lightweight dump，记录每个 detection group 的：

- `bg_prob`
- `group_size`
- `top1 score before bg`
- `top1 score after bg`
- `margin`
- `comp_topk activated ratio`

### 建议样本

先只跑：

1. MOT20-02，bg on
2. MOT20-02，bg off
3. MOT17-10，bg on

### 结构化记录要求

输出到：

- `outputs/haca_bg_diagnostic_<ts>/metrics.jsonl`
- `outputs/haca_bg_diagnostic_<ts>/summary.csv`

`summary.csv` 至少包括：

- `dataset`
- `seq`
- `bg_mode`
- `mean_bg_prob`
- `p90_bg_prob`
- `mean_group_size`
- `mean_margin`
- `suppressed_ratio`
- `status`

### 成功标准

至少确认以下任一条：

1. MOT20 bg-on 的 `bg_prob` 分布明显右移。
2. 大 group / 小 margin 样本上的 `s_prebg -> s_final` 压制异常大。
3. competition head 在高 bg_prob 下几乎被抵消。

### 继续条件

- B1 完成后，B3 才有明确修改方向。

---

## B2. HACA v1 / v2 / v3 的 no-background 矩阵评估

### 目的

确认 `v3` 的真实边际收益，而不是默认把最新版本当成最好 carrier。

### 涉及文件

- [scripts/run_haca_v3_mot20_train.sh](/gemini/code/FMtrack-main/FM-Track/scripts/run_haca_v3_mot20_train.sh:200)
- [scripts/run_nobg_mot20_eval.sh](/gemini/code/FMtrack-main/FM-Track/scripts/run_nobg_mot20_eval.sh:1)

### 要做什么

为 `mot20_haca_v1.npz / mot20_haca_v2.npz / mot20_haca_v3.npz` 分别跑统一 no-background baseline。

只需要 baseline，不需要同时带 Stage1/freeze。

### 结构化记录要求

建议目录：

- `outputs/haca_mot20_nobg_matrix_<ts>/summary.csv`

字段：

- `variant`
- `checkpoint`
- `HOTA`
- `AssA`
- `IDF1`
- `MOTA`
- `IDSW`
- `status`

### 成功标准

明确回答：

1. `v3` 是否稳定优于 `v2`。
2. `competition head` 在 no-bg 条件下是否确实提供净收益。
3. 如果 `v2 ~ v3`，则后续论文主线应该更聚焦“robust no-bg residual correction”，而不是复杂化 Stage1。

---

## B3. background gate 的 inference-side 轻量修复

### 目的

在不重训整个模型的前提下，测试 bg gate 是否能通过 runtime 轻量约束恢复一部分可用性。

### 候选修复

优先按从便宜到昂贵的顺序试：

1. `bg clamp`
   - `bg = min(bg, 0.3 / 0.5 / 0.7)`
2. `thresholded suppress`
   - 只有 `bg > tau` 才应用压制
3. `decoupled competition`
   - competition head 使用 `s_prebg`，最终抑制只在最后一步作用
4. `OOD fallback`
   - 当 `group_size` 或 `margin` 超过异常阈值时，直接退化到 `bg=0`

### 执行方式

先选单序列：

- MOT20-02

先跑 4-6 个点，而不是全 sweep。

### 结构化记录要求

建议目录：

- `outputs/haca_bg_patch_sweep_<ts>/summary.csv`

字段：

- `patch_name`
- `tau_or_cap`
- `seq`
- `HOTA`
- `AssA`
- `IDF1`
- `IDSW`
- `status`

### 成功标准

1. 至少能显著高于坏 baseline。
2. 最好能接近 no-bg probe 的 `HOTA=67.083`。
3. 如果所有轻量修复都明显不行，就停止在 bg 线上继续花时间。

### 继续条件

- 如果 B3 无法接近 no-bg，后续默认不再把 bg gate 作为主线修复目标。

---

## C1. Stage1 upper bound / headroom 分析

### 目的

回答一个比“再训一次 Stage1 会不会变好”更关键的问题：`MOT20 no-bg baseline` 还剩多少可被 Stage1 纠正的空间。

### 涉及文件

- [scripts/run_rgsa_stage1_mot20_nobg_train.sh](/gemini/code/FMtrack-main/FM-Track/scripts/run_rgsa_stage1_mot20_nobg_train.sh:1)
- [scripts/train_rgsa_stage1.py](/gemini/code/FMtrack-main/FM-Track/scripts/train_rgsa_stage1.py:1)
- `build_rgsa_labels.py` 生成的 label 数据

### 要做什么

对现有 label / oracle dump 做离线统计：

1. `accept / defer / reject` 比例
2. 真正能被 defer 后恢复的样本比例
3. 按 margin / group_size / seq 拆分的 recover ceiling

### 结构化记录要求

建议目录：

- `outputs/rgsa_stage1_headroom_<ts>/summary.csv`
- `outputs/rgsa_stage1_headroom_<ts>/bucket_metrics.csv`

至少输出：

- `total_pairs`
- `accept_count`
- `defer_count`
- `reject_count`
- `recoverable_count`
- `recoverable_ratio`

### 成功标准

如果 `recoverable_ratio` 很低，例如仍然只有千分级到低百分比，那么：

- Stage1 在 MOT20 上就不是当前高 ROI 方向。

---

## C2. Stage1 cheap sweep

### 目的

只在 C1 证明“确实有空间”之后，再用最便宜的方式试一轮 Stage1 调参。

### 建议只扫这几项

1. `margin-threshold`
2. `topk`
3. `focal-gamma`
4. `lambda-defer`

不要一开始就重写整个 Stage1 架构。

### 建议实验矩阵

最多 4-6 个点，例如：

| exp_id | margin_threshold | topk | focal_gamma | lambda_defer |
|---|---:|---:|---:|---:|
| C2a | 0.05 | 5 | 0.0 | 0.15 |
| C2b | 0.03 | 5 | 0.0 | 0.15 |
| C2c | 0.03 | 5 | 1.0 | 0.15 |
| C2d | 0.03 | 7 | 1.0 | 0.10 |

### 结构化记录要求

建议目录：

- `outputs/rgsa_stage1_mot20_cheap_sweep_<ts>/summary.csv`

### 成功标准

- 相对 no-bg baseline 至少达到 `HOTA +0.2` 才值得继续扩。
- 如果连这个阈值都过不了，就暂停 Stage1 主线。

---

## C3. freeze trigger audit

### 目的

确认 freeze 在 MOT20 上到底是“触发太少没价值”，还是“触发对象不对所以有轻微副作用”。

### 要做什么

对 no-bg baseline 和 `freeze_only` 运行做 match-level dump，统计：

- freeze rate
- 被 freeze 的样本分布
- 被 freeze 的样本后续是否真的避免了 IDSW

### 结构化记录要求

建议目录：

- `outputs/tcgau_mot20_audit_<ts>/summary.csv`
- `outputs/tcgau_mot20_audit_<ts>/frozen_cases.csv`

### 成功标准

如果发现：

1. freeze 几乎不触发，说明它在 MOT20 上不是有效控制杆；
2. freeze 触发了但多数样本不该冻，说明阈值语义不匹配。

不论哪种结果，都足够支持把 freeze 从 MOT20 主线后移。

---

## D1. MTCR / RuntimeReplay pilot

### 目的

只做小规模 pilot，判断更强的 runtime review 方案在 MOT20 上是否比 Stage1 更有 headroom。

### 执行方式

先只选单序列和少量 hard cases，不要全量训练。

推荐：

- MOT20-02
- MOT20-03

### 成功标准

- 只要 pilot 显示比 Stage1 更明显的 upper bound，就说明主研究精力可以转向 runtime replay / local conflict resolution。

---

## D2. RuntimeReplay upper bound

### 目的

做一个近似 oracle 的上界实验，回答“如果允许在高冲突局部重审，最多能拿回多少 HOTA / IDSW”。

### 价值

这个实验不一定直接变成产品逻辑，但非常适合做研究方向判断。

### 成功标准

- 如果上界明显高于当前 baseline，而 Stage1 上界很低，那么研究资源应该转向 replay / competition / local conflict 方向。

---

## E1. 效率基准

### 目的

补齐后续论文和工程决策都会问到的效率事实：

- 每帧耗时
- FPS
- GPU memory
- 额外参数量

### 建议比较对象

1. Laplace anchor only
2. HACA v2 no-bg
3. HACA v3 no-bg

### 结构化记录要求

建议目录：

- `outputs/haca_efficiency_benchmark_<ts>/summary.csv`

字段：

- `variant`
- `fps`
- `ms_per_frame`
- `gpu_mem_gb`
- `params_m`
- `status`

---

## E2. 泛化与论文补强

### 目的

在主线稳定后，再决定要不要补：

1. MOT17 no-bg 对照
2. DanceTrack 或其他 crowd / motion-heavy 数据集
3. competition head 可视化

这不是当前第一优先级，但它决定后续论文站位。

## 当前建议的实验顺序

建议实际执行顺序如下：

1. `A0` 记录系统修复 smoke test
2. `A1` 精确续跑当前被打断的 HACA 重训
3. `A2` pseudotrack 负载消融
4. `B1` bg_prob / group-size / margin 分布诊断
5. `B2` HACA v1 / v2 / v3 no-background 矩阵
6. `B3` bg gate inference-side 轻量修复
7. `C1` Stage1 upper bound / headroom 分析
8. `C2` Stage1 cheap sweep
9. `C3` freeze trigger audit
10. `D2` RuntimeReplay upper bound
11. `D1` MTCR / RuntimeReplay pilot
12. `E1` / `E2`

## 不建议现在优先做的事

下面这些方向现在不应该排在前面：

1. 直接大规模重写 Stage1 架构
2. 直接做全量 MTCR / RuntimeReplay 训练
3. 继续在已知崩坏的默认 bg baseline 上堆更多消融
4. 一开始就追求论文全故事，而不先把 no-bg 主线和记录系统跑稳

## Claude 执行任务清单

下面这部分是给 Claude 的可执行 checklist。目标是让 Claude 可以按顺序工作，不用再自己发明计划。

---

### 阶段 0：开工前检查

1. 读取以下文件，确认当前事实基线：
   - [docs/mot20_haca_experiment_roadmap_20260614.md](/gemini/code/FMtrack-main/FM-Track/docs/mot20_haca_experiment_roadmap_20260614.md:1)
   - [outputs/haca_mot20_train_20260613_082005/summary.csv](/gemini/code/FMtrack-main/FM-Track/outputs/haca_mot20_train_20260613_082005/summary.csv:1)
   - [outputs/haca_mot20_v3_nobg_full_eval_20260612_012822/summary.csv](/gemini/code/FMtrack-main/FM-Track/outputs/haca_mot20_v3_nobg_full_eval_20260612_012822/summary.csv:1)
   - [outputs/sca_lmf_mot20_nobg_eval_20260612_081934/summary.csv](/gemini/code/FMtrack-main/FM-Track/outputs/sca_lmf_mot20_nobg_eval_20260612_081934/summary.csv:1)
   - [outputs/sca_lmf_mot20_eval_20260609_081744/summary.csv](/gemini/code/FMtrack-main/FM-Track/outputs/sca_lmf_mot20_eval_20260609_081744/summary.csv:1)
2. 检查真实运行进程，不允许只看旧日志。
3. 明确当前没有实验在跑后，再开始改脚本或补跑。

通过标准：

- 形成一段简短状态摘要，指出：
  - 目前活跃主线是 no-bg HACA
  - 旧 bg baseline 已崩
  - Stage1/freeze 当前无收益

---

### 阶段 1：修记录系统

1. 阅读：
   - [scripts/append_experiment_record.py](/gemini/code/FMtrack-main/FM-Track/scripts/append_experiment_record.py:1)
   - [scripts/upsert_experiment_plan.py](/gemini/code/FMtrack-main/FM-Track/scripts/upsert_experiment_plan.py:1)
   - [scripts/run_nobg_mot20_followup.sh](/gemini/code/FMtrack-main/FM-Track/scripts/run_nobg_mot20_followup.sh:1)
   - [scripts/run_haca_v3_mot20_train.sh](/gemini/code/FMtrack-main/FM-Track/scripts/run_haca_v3_mot20_train.sh:1)
2. 完成最小修复：
   - queue 记录不再报错
   - 重跑时 plan/status 能从旧 failed 或 stale running 回到真实状态
   - 非零退出不再全部混成同一种失败语义
3. 产出：
   - 临时 smoke 目录 `outputs/_record_smoke_<ts>/summary.csv`
   - 更新后的 `outputs/experiment_plan.csv`
   - 更新后的 `outputs/experiment_registry.csv`
4. 汇报时说明：
   - 改了哪几个字段/脚本
   - smoke test 结果

通过标准：

- queue 级脚本可正常写 registry
- plan / summary / registry 三处对同一个 run 的状态一致

失败标准：

- 仍然出现 `kind=queue` 非法
- 手动重跑后状态仍保留旧 failed

---

### 阶段 2：续跑 HACA 重训

1. 阅读：
   - [outputs/haca_mot20_train_20260613_082005/train.log](/gemini/code/FMtrack-main/FM-Track/outputs/haca_mot20_train_20260613_082005/train.log:1)
   - [outputs/haca_mot20_train_20260613_082005/summary.csv](/gemini/code/FMtrack-main/FM-Track/outputs/haca_mot20_train_20260613_082005/summary.csv:1)
   - [scripts/run_haca_v3_mot20_train.sh](/gemini/code/FMtrack-main/FM-Track/scripts/run_haca_v3_mot20_train.sh:1)
2. 在脚本中加入：
   - 已有 checkpoint 的跳过逻辑
   - 重跑前即时状态回填
3. 用原 `OUT_ROOT=outputs/haca_mot20_train_20260613_082005` 续跑。
4. 确保产出：
   - `haca_v1/mot20_haca_v1.npz`
   - `haca_v2/mot20_haca_v2.npz`
   - `haca_v3/mot20_haca_v3.npz`
   - 更新后的 `summary.csv`

通过标准：

- shard 不被重复全量构建
- 训练链路顺利完成到 `haca_v3`
- 结构化记录显示 completed / success

失败标准：

- 仍然从 phase 0 全量重来
- summary 和实际产物不一致

---

### 阶段 3：做 pseudotrack 负载消融

1. 阅读：
   - [scripts/build_gt_pseudotrack_groups.py](/gemini/code/FMtrack-main/FM-Track/scripts/build_gt_pseudotrack_groups.py:118)
   - [scripts/run_haca_v3_mot20_train.sh](/gemini/code/FMtrack-main/FM-Track/scripts/run_haca_v3_mot20_train.sh:20)
2. 先实现一个小实验驱动脚本或队列脚本，固定：
   - `MOT20-01 train_half f0001_0120`
   - `MOT20-05 val_half f0001_0120`
3. 跑 4 组参数：
   - baseline
   - topk only
   - topk + hardneg
   - topk + hardneg + bigger batch
4. 为每组写入 `summary.csv`，记录 wall time、npz size、groups、candidates。

通过标准：

- 至少找到一组明显更轻的参数

失败标准：

- 所有组都几乎一样重
- 或者轻量参数导致候选覆盖崩塌

---

### 阶段 4：做 bg gate 分布诊断

1. 阅读：
   - [external/BoT-SORT-main/tracker/haca_assoc.py](/gemini/code/FMtrack-main/FM-Track/external/BoT-SORT-main/tracker/haca_assoc.py:455)
   - [external/BoT-SORT-main/tracker/bot_sort.py](/gemini/code/FMtrack-main/FM-Track/external/BoT-SORT-main/tracker/bot_sort.py:1237)
2. 增加轻量 debug dump，不要先做大重构。
3. 先跑：
   - MOT20-02 bg on
   - MOT20-02 bg off
   - MOT17-10 bg on
4. 输出：
   - `metrics.jsonl`
   - `summary.csv`
5. 汇报里必须给出：
   - bg_prob 分布对比
   - suppressed ratio
   - group_size / margin 与 bg_prob 的关系

通过标准：

- 能用数据解释 bg 崩坏，而不是只凭直觉

---

### 阶段 5：跑 HACA no-bg 版本矩阵

1. 基于同一训练产物，分别评估：
   - HACA v1 no-bg
   - HACA v2 no-bg
   - HACA v3 no-bg
2. 不带 Stage1，不带 freeze。
3. 输出到单独目录：
   - `outputs/haca_mot20_nobg_matrix_<ts>/summary.csv`

通过标准：

- 能明确说出当前最值得作为 carrier 的版本

---

### 阶段 6：判断 Stage1 还有没有 headroom

1. 读取现有 oracle dump 和 labels。
2. 做离线统计，不要先重新训练。
3. 输出：
   - `accept / defer / reject` 分布
   - recoverable ceiling
   - 按序列和 margin bucket 的可恢复比例

通过标准：

- 如果 headroom 很低，就明确暂停 Stage1 主线
- 如果确实有空间，再进入 cheap sweep

---

### 阶段 7：只在有 headroom 时做 Stage1 cheap sweep

1. 只扫：
   - `margin-threshold`
   - `topk`
   - `focal-gamma`
   - `lambda-defer`
2. 控制点数在 4-6 个以内。
3. 每组都要有独立结构化记录。

通过标准：

- 至少拿到 `HOTA +0.2`

失败标准：

- 最好点仍低于 no-bg baseline

---

### 阶段 8：freeze audit

1. 只审计，不先扩大 freeze 线。
2. 统计 freeze rate、冻结样本类型、冻结后是否减少 IDSW。

通过标准：

- 能判断 freeze 是“没触发”还是“触发对象错了”

---

### 阶段 9：RuntimeReplay / MTCR 只做 pilot

1. 先做 upper bound 或局部 pilot。
2. 不要上来做全量训练。

通过标准：

- 只要 upper bound 明显高于 Stage1，就说明后续研究重心可以转过去。

## 最后的执行纪律

Claude 在整个过程中必须遵守下面几点：

1. 每跑一个实验，都先定义输出目录和 `summary.csv`。
2. 每结束一个实验，都检查：
   - 真实进程
   - `summary.csv`
   - `experiment_plan.csv`
   - `experiment_registry.csv`
3. 不允许只凭日志认定实验状态。
4. 任何手动重跑，都要先回填结构化记录。
5. 每一轮汇报都要说清楚：
   - 这轮实验的目的
   - 结构化结果在哪里
   - 是否达到继续阈值

## 我的补充判断

综合几家模型的回复后，我认为最容易跑偏的地方有两个：

1. 把 `background gate 崩坏` 误解成 “整个 HACA v3 没学到东西”
2. 在 `Stage1 当前无收益` 的情况下，继续对 Stage1 过度投入

更稳的策略是：

- 把 no-bg HACA 当成当前真实 baseline；
- 用 B1/B2/B3 把 HACA 主体解释清楚；
- 只在 C1 证明还有 headroom 的前提下，再继续 Stage1。

如果后面 B2 发现 `v2 ~ v3`，而 C1 又显示 Stage1 headroom 很低，那么研究重点就应该更果断地转向：

- competition head 的真实贡献
- local conflict resolution
- runtime replay 上界

而不是继续维护一个在 MOT20 上没有净收益的多阶段 deferral 故事。
