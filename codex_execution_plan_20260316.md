# Codex 执行计划（2026-03-16）

## 0. 最终结论（先说死）

我已经把当前方向想清楚了：**没有新的补充会改变主判断**。

最终判断维持不变：

1. **当前 official MOT17 test 上，current association module 还不够有效。**
   - official 上 `base > heuristic > learned` in HOTA。
   - heuristic / learned 都没有超过 base。
   - DetA 基本相同，主要问题在 AssA 没有守住。
   - 因此，association-only 目前**不能继续作为主提交路线**。

2. **现在应该明确 pivot 到 detector / appearance 主链。**
   - 先把 host baseline 拉强。
   - 再在强 host 上验证 association 插件是否仍有真实增益。

3. **association 的定位要改成“强主链上的安全歧义关联插件（safe ambiguity-aware plug-in）”。**
   - 不是单独救活系统的 main engine。
   - 而是只在 hard ambiguous groups 上做 bounded residual refinement。

4. **old current learned 路线应该降级。**
   - 主线保留 `base` 和 `heuristic`。
   - learned 只保留 HACA-v3-safe 这种 bounded / ambiguity-triggered 版本作为 research plug-in。
   - old pair-wise calibrator 风格的 `current_learned` 不再作为主实验主角。

---

## 1. 这次没有改变方向的“补充点”是什么

没有新的补充会改变方向，但有 4 个执行层面的补充必须明确写入计划：

### 补充点 A：先锁死 protocol，再改模型
**为什么：**
MOTChallenge 官方明确建议在训练集上调参，并在 provided detections 上报告结果以保证可比性；官方 evaluation server 也不能被当作训练工具使用。现在只要 protocol 没锁死，detector gain、submit gain、association gain 就会混在一起。

**落实要求：**
- public / provided-det 线与 private-det 线必须彻底分开。
- 每次运行必须保存 manifest（配置、权重、阈值、commit、zip md5）。
- 任何“official 回升”都必须能追溯到唯一 profile。

### 补充点 B：这轮优先做 ReID-first host recovery
**为什么：**
BoT-SORT、StrongSORT、ByteTrack 这些强 tracking-by-detection 工作都说明：真正决定系统下限和大盘的，是 detector、appearance、score/lifecycle、association 一起构成的 host，而不是单一 learned matcher。当前仓库下，ReID 是最小改动、最高性价比、最容易 clean attribution 的升级点。

**落实要求：**
- 先升级 ReID backbone。
- 再做 lifecycle / threshold recovery。
- learned association 暂不作为本轮主变量。

### 补充点 C：association 以后只能按 safe plug-in 的逻辑改
**为什么：**
StrongSORT 证明 lightweight / plug-and-play refinement 是社区接受的叙事；GeneralTrack 又把 cross-scenario generalization 提到了核心位置。你当前最值钱的不是“再学一个分数”，而是：
- 只在 ambiguity 时触发；
- 只处理竞争组；
- 只做 bounded residual；
- 不破坏 easy cases。

### 补充点 D：必须建立 stop/go 判据，防止 association-only 再消耗主时间
**为什么：**
现在最危险的不是“方向不清楚”，而是“继续在 learned association 上惯性迭代”。

**硬判据：**
- 如果 stronger host 先把 official-like / official 分数明显拉回，而 association 仍不稳定，则 association-only 终止主线资格。
- 如果在 stronger host 上 heuristic / HACA-v3-safe 才开始稳定提升 AssA/HOTA，则 association 作为插件线保留。
- 如果 HACA-v3-safe 无法在第二 host 上复现，就降级为附录实验。

---

## 2. 顶会潜力现在变成什么样

### 2.1 负面判断
如果继续把论文写成“association-only 主提分方法”，那主会潜力已经很低。

**原因：**
- official 上 current association module 没有过 base。
- HOTA 是同时衡量 detection、association、localization 的综合指标。
- 在这个指标下，如果 association 模块 official 上不能守住整体 HOTA，它就很难撑起主贡献。

### 2.2 正面判断
如果改写成：

**“strong tracking-by-detection host + safe ambiguity-aware association plug-in”**

那么顶会潜力还在，而且是合理的。

### 2.3 这个潜力依赖什么
接下来是否还能冲主会，不取决于你能不能把 old learned 再调高一点，而取决于你能不能做出下面这组三联证据：

1. **强 host 确实能把 official / official-like 大盘拉回。**
2. **association 插件在强 host 上仍能带来额外 AssA / ID consistency 增益。**
3. **这种增益在第二 host / transfer setting 下仍成立。**

只要这三点做出来，文章就不是“调 matcher”，而是“面向强 MOT 系统的安全竞争关联插件”。

---

## 3. 给 Codex 的总执行原则

1. **先修 host，再碰 learned association。**
2. **先 public/provided-det clean line，再谈 private-det ceiling。**
3. **先建立唯一可复现 baseline profile，再做任何 ablation。**
4. **主实验不再把 old current_learned 当核心分支。**
5. **association 模块只按 plug-in 逻辑继续保留。**
6. **所有新增实验都必须带 run manifest。**
7. **所有实验都必须有结构化记录文件。**
   - 单实验至少要有 `result.csv` / `metrics.csv` / `*.metrics.jsonl` 之一。
   - 队列实验必须维护队列级 `summary.csv`。
   - 关键实验必须同步写入 `outputs/experiment_registry.csv`。
   - 运行中实验必须写明 `status=running`，不能让旧的 `failed` 状态残留误导判断。

---

## 4. 分阶段完整执行计划

# 阶段 1：锁死 protocol / baseline chain

## 4.1 目标
建立唯一可复现的 control baseline，彻底消除“结果来自不同 detector / 不同 submit chain / 不同阈值漂移”的不确定性。

## 4.2 为什么先做这个
如果这一步不先完成，后续 detector、ReID、association 的增益全部不可归因。

## 4.3 要改什么

### A. 在 `external/BoT-SORT-main/tools/track.py` 增加统一 profile 入口
新增类似：
- `--exp-profile mot17_public_stronghost_base`
- `--exp-profile mot17_public_stronghost_heuristic`
- `--exp-profile mot17_private_stronghost_base`

profile 内统一固定：
- detector source
- FastReID config
- ReID weights
- with_reid
- cmc method
- `track_high_thresh`
- `track_low_thresh`
- `new_track_thresh`
- `match_thresh`
- `track_buffer`
- `proximity_thresh`
- `appearance_thresh`
- assoc mode
- HACA checkpoint path

### B. 把 shell 脚本全部改成只接受 profile 名
把当前 submit / eval / official control 脚本改成：
- 不直接在命令行散着传阈值
- 统一由 profile 驱动

### C. 每次运行自动保存 `run_manifest.json`
至少记录：
- git commit
- profile 名
- detector path/hash
- reid config
- reid weight hash
- 所有 thresholds
- assoc mode
- output dir
- submit zip md5

## 4.4 交付物
- `configs/profiles/*.yaml` 或等价 json
- 改好的 `track.py`
- 改好的 eval / submit 脚本
- 一次 control run 的 manifest 样例

## 4.5 通过标准
- 同一个 profile 两次运行结果一致
- public/private 两条线不再混用
- 所有 official submit 都能追溯到 manifest

---

# 阶段 2：做 ReID-first host recovery（本轮主改）

## 5.1 目标
先把 host baseline 拉强，优先改善 AssA / ID consistency，并恢复 stronger host 的整体 HOTA 表现。

## 5.2 为什么是 ReID first，而不是 detector first

### 原因 1：改动最小
当前仓库已经接 FastReID，替换 appearance backbone 成本最低。

### 原因 2：归因最干净
先换 ReID，比同时换 detector 和 association 更容易看清谁在起作用。

### 原因 3：最贴近你的当前短板
official 上 DetA 变化很小，AssA 是更明显的短板，所以先打 appearance / association 接口更合理。

## 5.3 具体执行

### A. 默认 ReID 从 `SBS(S50)` 升到更强模型
优先级：
1. `SBS(R101-ibn)`
2. `SBS(R50-ibn)`
3. `AGW(S50)`

### B. 同步做 lifecycle / threshold recovery
要 sweep 的参数：
- `track_high_thresh`
- `track_low_thresh`
- `new_track_thresh`
- `match_thresh`
- `track_buffer`
- `proximity_thresh`
- `appearance_thresh`

### C. 这轮不把 learned association 当主变量
只跑：
- stronger host + base
- stronger host + heuristic

HACA-v3-safe 放到下一阶段。

## 5.4 要改什么文件
- `external/BoT-SORT-main/tools/track.py`
- 相关 ReID config 引用
- eval / submit 脚本
- 最好新增 `configs/profiles/mot17_public_reid_upgrade.yaml`

## 5.5 交付物
- stronger ReID host 的 profile
- lifecycle sweep 日志和 summary 表
- stronger host 的 val 结果
- 一次 clean official-like / official 提交流程

## 5.6 通过标准
- stronger host 至少在 val 上稳定优于 current base
- transfer 不崩
- official-like / submit chain 没有配置漂移

## 5.7 失败后的解释
如果 stronger host 都拉不起来，说明当前问题不只是 association，而可能是：
- protocol 仍未锁死
- detector / feature 输入链仍有问题
- lifecycle / score handling 未恢复

---

# 阶段 3：只在 protocol 锁死后，给 detector 单独开支路

## 6.1 目标
回答“当前瓶颈有多少来自 detection input”。

## 6.2 为什么 detector 不能抢先成为主线
因为一旦 detector 和 protocol 没分开，后面所有增益都不可解释；而且 provided/public comparable 线与 private-det ceiling 线必须是两套叙事。

## 6.3 具体执行

### A. 先做 detector result replacement，不重写 tracker 主体
只替换 detection input，保持 tracker 主体不动。

### B. 用外部 pluggable harness 快速扫组合
可先用 BoxMOT 风格的外部 harness 做 detector / ReID 的快速组合筛查，再决定是否回写主仓库。

### C. detector 候选优先级
如果允许 private-det 线：
1. RT-DETR / RT-DETRv2 级别 detector
2. 你已有生态里最好落地的 YOLOX / YOLO 强模型

## 6.4 交付物
- detector-only replacement 结果表
- public / private-det 两线的差异说明
- 是否值得把 stronger detector 合回主仓库的结论

## 6.5 通过标准
- stronger detector 的收益可解释
- DetA / HOTA 的变化明显
- 与 ReID-upgraded host 的相互作用关系清楚

---

# 阶段 4：重建 association 训练分布（host-aware candidate replay）

## 7.1 目标
让 learned association 看到和 runtime 更一致的 candidate competition 分布，而不是继续依赖旧的 same-base GT pseudo-track 假设。

## 7.2 为什么必须改
当前 same-base val 能涨、official 掉，最可能的主因之一就是训练分布和 runtime 分布不一致。

## 7.3 具体执行

### A. 改 `scripts/build_gt_pseudotrack_groups.py`
新增 `tracker_dump replay` 模式：
- 从 tracker runtime dump 出来的 candidate set 重建 group
- 记录 valid mask / top-k rivals / background / weak history
- 支持 stronger host 特征

### B. 每个 group 输出更多统计
至少加：
- ambiguity margin
- group size
- weak history flag
- background candidate ratio
- host tag
- tracker family tag

### C. 至少构建两类 shard
- BoT-SORT host shard
- StrongSORT-style host shard

## 7.4 为什么要多 host shard
因为接下来论文如果要有顶会潜力，就必须证明 learned plug-in 不只是一个 host-specific trick。

## 7.5 交付物
- 改好的 `build_gt_pseudotrack_groups.py`
- host-aware replay 数据集
- 分布统计 markdown / csv

## 7.6 通过标准
你能明确回答：
- hard groups 占比多少
- top1/top2 gap 分布如何
- background/null 竞争比例如何
- BoT-SORT 与 StrongSORT host 的 group 分布差多少

如果这些都答不出来，learned 训练暂不允许继续扩张。

---

# 阶段 5：彻底降级 old current_learned，只保留 heuristic 与 HACA-v3-safe

## 8.1 目标
把 association 研究线收敛到“安全插件”版本，不再浪费时间在 old pair-wise calibrator 上。

## 8.2 为什么这样做
old current_learned 更像已有相似度的局部重权重，而不是对竞争关系的真正建模；这条路的 official 风险已经充分暴露。

## 8.3 具体执行

### A. 改 `tracker/bot_sort.py`
主实验矩阵只保留：
- `base`
- `heuristic`
- `haca_v3_safe`

`current_learned` 仅保留为 archival / debug 对照，不再出现在主线实验表里。

### B. 改 `tracker/haca_assoc.py`
继续强化下面这些约束：
- 只在 ambiguity-triggered groups 触发
- 只改 top-k rivals
- bounded residual
- background/null 保持 base path
- easy groups 接近零改动
- 尽量 zero-sum / no drift

### C. 改 `scripts/train_haca_v3_from_gt_tracks.py`
训练逻辑保留：
- freeze base scorer
- 先训 competition head
- ambiguous groups oversampling
- safe regularization

但新增：
- multi-host shard mixing
- stronger host feature support
- host tag conditioning（如果实现成本可控）

## 8.4 交付物
- 收敛后的 association mode registry
- 改好的 HACA-v3-safe 训练脚本
- 新 checkpoint

## 8.5 通过标准
HACA-v3-safe 只有在同时满足下面三点时才保留为正文方法：
1. strongest host 上优于 heuristic
2. second host 上不崩
3. hard groups activation 显著高于 easy groups

否则，降级为附录实验。

---

# 阶段 6：实验只排 3 组，不再发散

## 实验 1：Host recovery
**目的：**回答“更强 detector / ReID 是否能明显拉高主链表现”。

### 最小版本
- current base host
- ReID-upgraded base host
- ReID-upgraded + tuned lifecycle host

### 如果允许 private-det，再加一个可选分支
- stronger detector + current / stronger ReID host

### 想回答的问题
- 主瓶颈是不是 host
- ReID-first 是否能显著改善 AssA / HOTA
- lifecycle tuning 是否是必要配套

---

## 实验 2：Strongest host 上的 plugin value
**目的：**回答“association 插件在强 host 上是否仍有价值”。

### 比较对象
- strongest host + base
- strongest host + heuristic
- strongest host + HACA-v3-safe

### 想回答的问题
- heuristic 在强 host 上是否仍稳
- HACA-v3-safe 是否超过 heuristic
- 插件的增益是 AssA、IDSW 还是总体 HOTA

---

## 实验 3：Cross-host / transfer sanity
**目的：**回答“association-only 是否还值得继续，以及插件是否有泛化价值”。

### 比较对象
在第二 host 上跑：
- strongest host counterpart + heuristic
- strongest host counterpart + HACA-v3-safe

### 想回答的问题
- learned plug-in 是否是 host-specific trick
- 如果换 host 仍稳定，才有资格保留为论文主方法之一

---

# 阶段 7：论文叙事的最终改法

## 10.1 你不再讲什么
不再讲：
- learned association 是主提分路线
- association-only 可以直接拉起 system-level result
- main novelty 是“学一个更强分数”

## 10.2 你改成讲什么
改成讲：

**“We build a strong tracking-by-detection host first, then add a safe ambiguity-aware association refinement that only intervenes on hard competitive groups.”**

中文就是：

**“先建立强主链，再在高歧义竞争组上加入安全的受限关联修正插件。”**

## 10.3 贡献建议写法

### 贡献 1
建立一个 protocol-clean、appearance-strong 的 tracking-by-detection host。

### 贡献 2
提出一个 ambiguity-triggered、competition-aware、bounded residual 的 association plug-in。

### 贡献 3
证明该 plug-in 在强 host 上改善 identity consistency，并在第二 host 上保持稳定。

## 10.4 为什么这是更好的顶会 framing
因为它同时满足：
- 有清晰 host baseline
- 有清晰 method boundary
- 有 hard-case motivation
- 有 safety / generalization evidence
- 不与当前 official 失败证据冲突

---

## 5. 让 Codex 直接执行的优先级顺序

### P0（立刻做）
1. 给 `track.py` 增加 profile 机制
2. 改 eval / submit 脚本为 profile 驱动
3. 加 run manifest
4. 切出 public / private-det 两条 clean line

### P1（本轮主改）
5. 升级 ReID：先 `SBS(R101-ibn)`，再备选 `SBS(R50-ibn)` / `AGW(S50)`
6. 做 lifecycle / threshold sweep
7. 只跑 stronger host + {base, heuristic}

### P2（确认 host 拉起后再做）
8. 做 detector result replacement 支路
9. 判断是否值得并入主仓库

### P3（research plugin 线）
10. 改 `build_gt_pseudotrack_groups.py` 为 host-aware replay
11. 重训 HACA-v3-safe
12. 在 strongest host + second host 上验证 plugin

### P4（论文整理）
13. 改 one-pager / paper outline / contribution wording
14. 删除 old current_learned 作为主表主线
15. 重写方法定位：safe plug-in on strong host

---

## 6. 明确的停损条件（必须执行）

### 停损条件 1
如果 stronger host 都拉不动 official-like / official，大概率说明 protocol / detector / lifecycle 仍没锁死，禁止继续在 association-only 上投入。

### 停损条件 2
如果 HACA-v3-safe 在 strongest host 上也过不了 heuristic，禁止继续把 learned 写成主贡献。

### 停损条件 3
如果 HACA-v3-safe 在第二 host 上明显崩溃，正文降级，最多保留为附录分析。

### 停损条件 4
如果 detector/private-det 引入后 public comparable 叙事变脏，则必须把两条结果表彻底拆开，禁止混写。

---

## 7. 最终一句话（给 Codex 和后续自己看的）

**现在不是继续证明 association 能不能单独救系统的时候；现在是先把系统主链拉回强 baseline，然后再证明你的 association 是强系统上的额外真贡献。**

也就是说：

- **主线：强 host 恢复（优先 ReID-first）**
- **辅线：heuristic 保留**
- **研究插件线：HACA-v3-safe**
- **暂停主投入：old current_learned / association-only 继续深挖**

---

## 8. 外部依据（供后续写 paper / README 时参考）

- MOTChallenge instructions：建议在 training set 上调参，并在 provided detections 上报告结果以保证可比性；evaluation server 不能用于训练。
- HOTA：是同时衡量 detection、association、localization 的综合 MOT 指标，因此单一 association 改动如果 official 上守不住整体 HOTA，就很难成为主贡献。
- BoT-SORT：强 tracking-by-detection 系统是 motion、appearance、CMC、Kalman 等主链共同做强，而不是依赖单一匹配头。
- StrongSORT：强 baseline + lightweight plug-and-play refinement 是被社区接受的路径。
- ByteTrack：detection score/lifecycle 的恢复本身就是主链关键，不是附属调参。
- FastReID model zoo：当前仓库已经具备低成本切换更强 ReID backbone 的条件。
- GeneralTrack：generalizability 已经是高水平 MOT 工作的重要评价维度。
- LA-MOTR：learnable association 能发，但前提是 association 本身是清晰、完整、可验证的设计，而不是一个弱小的 pair-wise calibrator。
