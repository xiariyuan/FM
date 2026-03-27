# Pro Review Reply: Stronger V2 Design While Large-Data Run Is In Flight

日期: 2026-03-25

对应提示词:

- `md/PRO_REVIEW_SEND_TO_PRO_STRONGER_V2_WHILE_LARGEDATA_RUNNING_20260324.md`

## 管理级裁决

- 判定: `NARROW GO`
- 近期不要再补 `v13` portability check。
- 近期唯一主方案: `B. larger training data`
- 中期唯一更强模块: `HostConditionedLocalConflictSetPredictor`

## Pro 核心回答摘要

- `v15` stronger-host negative 已经足够说明 current tiny-data / `v1` checkpoint 不具备 zero-shot portability。
- 当前最值钱的一枪不是再赌一个 host，而是先把 base-host 数据面扩大并重训。
- 在 operator 语义保持不变的前提下，下一代更强模块应从 `LocalConflictCommitRefiner v1` 升级成一个:
  - host-conditioned
  - local conflict cluster level
  - row/column-aware
  - set-prediction 风格
  - conservative partial commit + defer-to-host
  的 `v2` 模块。

## Pro 给出的 v2 设计方向

主方案:

- `HostConditionedLocalConflictSetPredictor`

备选:

- `LocalConflictEdgeMPNRefiner-v2`

不建议优先走的路线:

- 仅仅加宽加深当前 MLP
- 直接改成整机 `TrackFormer / MOTR / MOTIP`
- 当前阶段就改成 `Sinkhorn / OT` 解码
- 回到 row-local / full replacement / continuity

## Pro 明确要求补强的三件事

1. 特征语义

- 不再只吃 raw host score。
- 增加 row-normalized / column-normalized score features。
- 增加 geometry delta / IoU / bbox distance。
- 增加 host conditioning / host embedding。

2. loss 设计

- 主损失保留 `[local tracks + defer]` assignment CE。
- 新增 edge-level auxiliary loss。
- 新增 cluster-level safety / gate loss。
- 新增 conservative margin loss，强化 commit vs defer 的安全边界。

3. 结构表达

- 不再只做 edge-MLP + pooling。
- 显式建模:
  - same-row competition
  - same-column collision
  - cluster-level safety gate

## 建议的 first-priority experiment

- 先在 larger-data base-host 数据面上训练 `v2`。
- 先只打 `base_reid_da` 上的 paired proxy0213。
- 只在 `v2` 先赢当前 enlarged-data `v1` 后，再回 stronger-host portability。

## 我们采纳的动作

- 不再补 `v13` portability check。
- 直接实现 `set_predictor_v2` 完整代码链，不做最小 MVP。
- runtime 继续保持:
  - `primary-only`
  - `pre-Hungarian`
  - `partial commit + defer to host`
- 训练链升级为:
  - v2 dataset builder
  - v2 trainer
  - v2 proxy runner
  - v2 generic runner
  - v2 large-base stage1 queue

