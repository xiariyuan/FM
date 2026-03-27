# 完整版 Learned Runtime 实现说明

日期：2026-03-18

## 1. 这次落地的不是最小版

当前已经新增的不是“表格特征 MLP”，而是完整的 runtime-replay learned reranker 训练链，包含：

1. `scalar evidence tower`
2. `temporal evidence tower`
3. `candidate competition tower`
4. `safety / activation controller`
5. `raw runtime tensor dump -> labeled replay -> trainable shard -> learned training` 全链路

---

## 2. 已实现文件

### 2.1 模型

- [models/runtime_replay_assoc.py](/gemini/code/FMtrack-main/FM-Track/models/runtime_replay_assoc.py)

核心内容：

- `RuntimeReplayAssociationAdapter`
- scalar tower
- multi-scale temporal tower
- set attention + duel aggregation
- group activation / null / delta cap
- bounded residual output
- `save_runtime_replay_checkpoint()`

### 2.2 原始 runtime 张量导出

- [models/runtime_tracker_bytetrack.py](/gemini/code/FMtrack-main/FM-Track/models/runtime_tracker_bytetrack.py)

新增能力：

- 继续保留原 CSV dump
- 新增 raw tensor shard dump
- 导出内容包括：
  - `group_ids`
  - `group_offsets`
  - `det_feat`
  - `det_box`
  - `det_score`
  - `cand_track_rank`
  - `cand_track_id`
  - `cand_hist_feat`
  - `cand_hist_mask`
  - `cand_hist_time`
  - `cand_track_box`

### 2.3 构建训练 shard

- [scripts/build_runtime_assoc_group_shards.py](/gemini/code/FMtrack-main/FM-Track/scripts/build_runtime_assoc_group_shards.py)

作用：

- 把 `labeled replay CSV` 和 `raw tensor shards` 对齐
- 生成可直接训练的 `runtime_replay_shard_*.npz`
- 可选写入 teacher score

### 2.4 完整训练脚本

- [scripts/train_runtime_replay_reranker.py](/gemini/code/FMtrack-main/FM-Track/scripts/train_runtime_replay_reranker.py)

当前支持：

- 按 group 训练
- `top-k + force include positive`
- listwise CE
- hard duel loss
- safe no-op loss
- distillation loss
- gate regularization
- 每个 epoch 输出：
  - `base_top1`
  - `final_top1`
  - `top1_gain`
  - `amb_top1_gain`
  - `easy_shift`
  - `bg_suppression`
  - `recoverable_rate`
  - `active_rate`

### 2.5 一键长跑脚本

- [scripts/run_complete_runtime_replay_pipeline.sh](/gemini/code/FMtrack-main/FM-Track/scripts/run_complete_runtime_replay_pipeline.sh)

作用：

- dump runtime candidates
- label replay groups
- build trainable shards
- train full learned runtime

---

## 3. 当前架构细节

### 3.1 输入对象

每个 group 是：

- 一个 detection
- 它当前 runtime primary association 里的 candidate tracks

### 3.2 scalar tower 输入

默认复用当前 GBDT 已证明有效的特征：

- `anchor_score`
- `base_score`
- `refined_score`
- `motion_score`
- `det_score`
- `log1p_track_gap`
- `log1p_track_hist_len`
- `base_margin`
- `refined_margin`
- `rank_margin`
- `rank_entropy`
- `rank_frac`
- `dx_norm`
- `dy_norm`
- `log_w_ratio`
- `log_h_ratio`
- `log_area_ratio`
- `det_track_iou`

### 3.3 temporal tower 输入

- detection feature
- candidate history feature sequence
- history mask
- history time

中间显式构造：

- det-history cosine similarity
- similarity delta
- age / time feature
- valid mask

### 3.4 competition tower

不是 mean/max group context。

当前实现是：

- set self-attention
- pairwise duel token
- duel attention aggregation

### 3.5 safety controller

当前 group-level controller 输入包括：

- top1-top2 margin
- entropy
- candidate count
- history sufficiency
- temporal confidence
- temporal uncertainty
- det score

输出包括：

- `group_activation`
- `null_logit`
- `delta_cap`

---

## 4. 训练目标

训练脚本目前使用：

1. `listwise CE`
2. `hard duel loss`
3. `safe no-op loss`
4. `teacher distillation loss`
5. `gate regularization`

这比旧的 GT pseudo-group + pair-wise calibrator 明显更接近真实 runtime object。

---

## 5. 现在怎么启动

### 5.1 一键长跑

```bash
bash /gemini/code/FMtrack-main/FM-Track/scripts/run_complete_runtime_replay_pipeline.sh
```

### 5.2 指定 teacher / detector / mode

```bash
TEACHER_MODEL=/path/to/model.pkl \
DETECTOR=sw_yolox \
MODE=base \
SCOPE=full7 \
TRAIN_DEVICE=cuda \
TRAIN_EPOCHS=15 \
bash /gemini/code/FMtrack-main/FM-Track/scripts/run_complete_runtime_replay_pipeline.sh
```

---

## 6. 当前还没有做的事

现在还没有把这个完整 learned runtime 正式接回 online runtime primary association 做最终 on/off eval。

原因不是不能接，而是当前更重要的顺序是：

1. 先确认完整 learned runtime 在 offline replay 上是否至少追平或超过 GBDT
2. 再把它接回 frozen host 做 online plugin on/off

这个顺序更稳，也更符合你当前证据链需要。

---

## 7. 下一步最关键的判断标准

不是只看 loss。

要重点看：

1. `top1_gain`
2. `amb_top1_gain`
3. `easy_shift_mean`
4. `recoverable_rate`
5. learned 是否能追平或超过 GBDT baseline

如果完整 learned runtime 仍然明显打不过 GBDT：

- 说明真正的问题不是“模型太小”
- 而是 runtime replay object 上，结构化表格边界已经很强，深模型需要进一步明确比 GBDT 多吃到的时序信息

如果它能稳定超过 GBDT：

- 才值得进入 online integration 和 host on/off eval 阶段
