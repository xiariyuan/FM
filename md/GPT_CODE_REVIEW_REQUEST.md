# FM-Track: 频域感知多目标跟踪 - 顶会级代码审查请求

## 投稿目标

CVPR / ICCV / ECCV (计算机视觉顶会)

---

## 核心创新点

我们提出 **FM-Track**，一个频域感知的多目标跟踪框架，核心创新包括：

### 1. Learnable Frequency Decomposition (LFD)
- 首次在 MOT 中引入可学习的频率分解
- 将轨迹特征分解到多个频带，不同频带捕获不同时间尺度的运动模式
- 低频 → 长期身份信息；高频 → 短期运动变化

### 2. Frequency-Temporal Transformer (FTT)
- 分频带时序建模 + 跨频带注意力融合
- 低频用 Mamba（长依赖）；高频用 Transformer（短期交互）

### 3. Frequency-Aware ID Decoder
- 双分支设计：标准分支 + 频率分支
- 基于频带置信度的自适应融合

### 4. 检测器无关设计
- 冻结预训练检测器（ByteTrack/YOLOX），只训练关联模块
- 证明频域关联本身的有效性，而非依赖更强的检测器

---

## 当前实现：ByteTrack 特征提取 + 频域关联

### 架构图

```
ByteTrack (YOLOX) - 冻结
    │
    ├─→ Backbone (PAFPN) → FPN 特征图 [P3, P4, P5]
    │
    ├─→ Head (stems) → 统一通道特征 (256*width)
    │
    └─→ RoIAlign → 检测框特征 (N, 256)
                      │
                      ↓ Projection Layer（可训练）
                      │
         ┌────────────┴────────────┐
         │                         │
    Trajectory Features       Detection Features
         │                         │
         └──────────┬──────────────┘
                    ↓
    ┌─────────────────────────────────────┐
    │  Learnable Frequency Decomposition  │
    │  → K frequency bands [F0...FK-1]    │
    └─────────────────────────────────────┘
                    ↓
    ┌─────────────────────────────────────┐
    │  Frequency-Temporal Transformer     │
    │  - Low-freq: Mamba (long-range)     │
    │  - High-freq: Attention (short)     │
    │  - Cross-band interaction           │
    └─────────────────────────────────────┘
                    ↓
    ┌─────────────────────────────────────┐
    │  Frequency-Aware ID Decoder V2      │
    │  - Standard branch (Mamba + Attn)   │
    │  - Frequency branch (per-band head) │
    │  - Confidence-weighted fusion       │
    └─────────────────────────────────────┘
                    ↓
              ID Prediction
```

---

## 需要审查的新代码（5个文件）

### 文件清单

| 文件 | 功能 | 行数 |
|------|------|------|
| `models/bytetrack_feature_extractor.py` | YOLOX 特征提取器，使用 RoIAlign 提取检测框特征 | ~350 |
| `train_bytetrack.py` | 训练脚本，冻结检测器训练关联模块 | ~450 |
| `configs/bytetrack_fa_mot_mot17.yaml` | MOT17 训练配置 | ~100 |
| `models/runtime_tracker_bytetrack.py` | 运行时跟踪器，在线 ID 关联 | ~300 |
| `submit_bytetrack.py` | 推理评估脚本 | ~250 |

---

## 审查维度（顶会标准）

### 1. 方法论正确性 [最重要]

- [ ] 特征提取流程是否符合论文描述？
- [ ] 频域分解的输入输出维度是否正确？
- [ ] 冻结检测器的实现是否正确？（确保没有梯度泄漏）
- [ ] 训练时使用 GT 框提取特征是否合理？（reviewer 可能质疑）

### 2. 实验公平性 [顶会审稿重点]

- [ ] 与 baseline 的比较是否公平？
  - 相同的检测器
  - 相同的训练数据
  - 相同的评估协议
- [ ] 是否存在信息泄露？
  - 训练集和测试集是否严格分开
  - 验证时是否使用了测试集信息

### 3. 代码正确性

- [ ] 坐标系转换是否正确？(xywh ↔ cxcywh, 像素 ↔ 归一化)
- [ ] RoIAlign 的 spatial_scale 是否正确？
- [ ] 边界情况处理（空检测、空轨迹、单目标）
- [ ] 张量维度在整个流程中是否一致？

### 4. 潜在的 Reviewer 质疑点

请特别检查以下可能被 reviewer 质疑的问题：

**Q1: 为什么训练时用 GT 框而不是检测框？**
- 当前实现是否会导致 train-test gap？
- 是否需要加入检测框噪声模拟？

**Q2: 特征提取方式是否最优？**
- RoIAlign from FPN P3 是否是最佳选择？
- 是否应该使用多尺度特征融合？

**Q3: 冻结检测器的设计合理性？**
- 如何证明是关联模块的贡献而非检测器的贡献？
- 消融实验设计是否完整？

**Q4: 与 SOTA 方法的公平比较？**
- ByteTrack 原始方法使用了 IoU + ReID
- 我们是否在相同设置下比较？

### 5. 消融实验支持

请检查代码是否支持以下消融实验：

- [ ] 不同频带数量 (K=2,4,8)
- [ ] 有/无 Mamba for low-freq
- [ ] 有/无 cross-band interaction
- [ ] 不同特征提取方式
- [ ] 投影层 vs 直接使用 YOLOX 特征

### 6. 可复现性

- [ ] 随机种子是否固定？
- [ ] 所有超参数是否在配置文件中？
- [ ] 是否有隐式的硬编码？

---

## 已有的频域模块接口（无需审查，但需理解）

```python
# FrequencyAwareTrajectoryModeling 输入输出
seq_info = {
    "trajectory_features": (B, G, T, N, C=256),  # 历史轨迹特征
    "trajectory_boxes": (B, G, T, N, 4),         # cxcywh 归一化
    "trajectory_masks": (B, G, T, N),            # True=padding
    "trajectory_id_labels": (B, G, T, N),        # 轨迹 ID
    "trajectory_times": (B, G, T, N),            # 时间戳
    "unknown_features": (B, G, TU, NU, C),       # 当前帧检测特征
    "unknown_boxes": (B, G, TU, NU, 4),
    "unknown_masks": (B, G, TU, NU),
    "unknown_id_labels": (B, G, TU, NU),         # GT ID（训练用）
    "unknown_times": (B, G, TU, NU),
}

# FrequencyAwareTrajectoryModeling 输出
# 返回更新后的 seq_info，添加：
#   - "freq_info": 频率分解信息
#   - "freq_losses": {"ortho_loss": ...}
#   - 更新的特征

# FrequencyAwareIDDecoderV2 输出
(id_logits, id_gts, id_masks, freq_extra_losses)
# id_logits: (num_layers, B, G, TU, NU, vocab_size)
# freq_extra_losses: {"consistency_loss": ..., "loss_weights": [...]}
```

---

## 期望输出格式

### Part 1: 严重问题（会导致论文被拒）

```
[CRITICAL] 文件:行号
问题: xxx
影响: 可能导致 xxx
修复: xxx
```

### Part 2: 方法论问题（reviewer 可能质疑）

```
[METHODOLOGY] 问题描述
潜在质疑: reviewer 可能问 xxx
建议回应/修改: xxx
```

### Part 3: 实现问题（影响结果但不致命）

```
[IMPL] 文件:行号
问题: xxx
建议: xxx
```

### Part 4: 消融实验建议

```
[ABLATION] 建议添加的消融实验
目的: 证明 xxx
实现方式: xxx
```

### Part 5: 总体评价

- 创新性评分 (1-10)
- 实现完整度评分 (1-10)
- 顶会录用可能性评估
- 最需要加强的方面

---

## 需要审查的代码文件

请审查以下 5 个文件：

1. `models/bytetrack_feature_extractor.py`
2. `train_bytetrack.py`
3. `configs/bytetrack_fa_mot_mot17.yaml`
4. `models/runtime_tracker_bytetrack.py`
5. `submit_bytetrack.py`

这些文件与本 markdown 文件位于同一项目目录中。

---

## 补充说明

### 项目结构

```
FM-Track/
├── models/
│   ├── bytetrack_feature_extractor.py  [待审查]
│   ├── runtime_tracker_bytetrack.py    [待审查]
│   ├── bytetrack_detector.py           [已有]
│   └── motip/
│       ├── freq_aware_trajectory_modeling.py  [核心模块，已有]
│       ├── freq_aware_id_decoder_v2.py        [核心模块，已有]
│       ├── learnable_freq_decomposition.py   [核心模块，已有]
│       └── freq_temporal_transformer.py      [核心模块，已有]
├── train_bytetrack.py                  [待审查]
├── submit_bytetrack.py                 [待审查]
├── configs/
│   └── bytetrack_fa_mot_mot17.yaml     [待审查]
├── third_party/
│   └── ByteTrack/                      [外部依赖]
└── weight/
    └── bytetrack_x_mot17.pth.tar       [预训练权重]
```

### 数据格式

- **检测框格式**：
  - `xywh`: 像素坐标，左上角 + 宽高
  - `cxcywh`: 归一化坐标，中心点 + 宽高，值域 [0,1]

- **特征维度**：
  - YOLOX stem 输出: `256 * width` (YOLOX-X: 320, YOLOX-L: 256)
  - 投影后: 256
  - 频域模块期望: 256

### 训练流程

```
1. 加载图像和标注
2. 从标注获取 GT 框 (cxcywh 归一化)
3. 转换为像素坐标 xywh
4. 使用 ByteTrack 特征提取器从 GT 框位置提取特征
5. 构造 seq_info 字典
6. 前向传播：trajectory_modeling → id_decoder
7. 计算损失：id_loss + freq_ortho_loss + freq_consistency_loss
8. 反向传播（只更新关联模块和投影层）
```

### 推理流程

```
1. ByteTrack 检测 + 特征提取
2. 过滤低置信度检测
3. 使用历史轨迹特征构造 seq_info
4. 频域关联模块预测 ID
5. 匈牙利/贪婪分配
6. 更新轨迹状态
7. 输出跟踪结果
```
