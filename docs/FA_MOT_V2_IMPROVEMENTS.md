# FA-MOT V2 改进说明

## 1. 当前状态分析

### 1.1 已完成的双分支架构

```
                    输入特征 (B, G, T, N, 256)
                            │
                            ▼
            ┌───────────────────────────────────┐
            │  FrequencyAwareTrajectoryModeling │
            │  ┌─────────────────────────────┐  │
            │  │ LFD (可学习频率分解)         │  │
            │  │ FTT (频率-时序Transformer)   │  │
            │  │ Global Mamba                │  │
            │  └─────────────────────────────┘  │
            └───────────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            │                               │
            ▼                               ▼
    ┌───────────────┐               ┌───────────────┐
    │ 频率分支 (FGA) │               │ 标准分支      │
    │ 多频带ID预测   │               │ Cross-Attn    │
    │ 遮挡感知      │               │ (简化版)      │
    └───────────────┘               └───────────────┘
            │                               │
            └───────────┬───────────────────┘
                        │
                        ▼
                   α=0.5 固定融合
                        │
                        ▼
                    输出ID预测
```

### 1.2 发现的问题

| 问题 | 描述 | 影响程度 |
|------|------|---------|
| **ID解码器简化过度** | 原版6层→简化版1层 | ⚠️ 高 |
| **融合权重固定** | α=0.5 硬编码 | ⚠️ 中 |
| **辅助损失未使用** | USE_AUX_LOSS=False | ⚠️ 中 |
| **相对位置编码缺失** | 简化版没有rel_pe | ⚠️ 中 |
| **形状不匹配时退化** | 直接回退单分支 | ⚠️ 低 |

---

## 2. V2版本改进

### 2.1 FrequencyAwareIDDecoderV2 架构

```
                    输入特征
                        │
        ┌───────────────┴───────────────┐
        │                               │
        ▼                               ▼
┌───────────────────┐           ┌───────────────────┐
│   频率分支 (新)    │           │   标准分支 (完整)   │
│                   │           │                   │
│ • 多频带ID预测头   │           │ • 6层解码器        │
│ • 频带融合网络    │           │   - Self-Mamba    │
│ • 一致性评估     │           │   - Cross-Attn    │
│ • 频率门控增强    │           │   - FFN           │
│                   │           │ • 相对位置编码     │
│                   │           │ • 辅助损失        │
└───────────────────┘           └───────────────────┘
        │                               │
        │           ┌───────────────────┘
        │           │
        ▼           ▼
┌───────────────────────────────┐
│   LearnableFusionWeight       │
│                               │
│ • 基于频率一致性动态调整         │
│ • 频率一致性作为输入          │
│ • 可学习的基础权重先验         │
└───────────────────────────────┘
                │
                ▼
            融合输出
```

### 2.2 关键改进点

#### 2.2.1 完整的6层解码器结构
```python
for layer in range(num_layers):  # num_layers=6
    # 1. Self-Mamba (层>0时)
    if layer > 0:
        out = self.self_mamba_layers[layer-1](embeds)
    
    # 2. Cross-Attention with Relative Position Encoding
    rel_pe_mask = self.rel_pos_embeds[layer][rel_pe_idxs]
    cross_out = self.cross_attn_layers[layer](query, key, value, attn_mask=mask+rel_pe)
    
    # 3. FFN
    out = self.ffn_layers[layer](cross_out)
    
    # 4. 每层输出预测（辅助损失）
    logits = self.embed_to_word_layers[layer](out)
```

#### 2.2.2 可学习的融合权重
```python
class LearnableFusionWeight(nn.Module):
    def forward(self, features, freq_consistency, freq_weights):
        # 基于频率一致性和特征内容预测融合权重
        input_feat = concat([features, freq_weights])
        weights = self.weight_predictor(input_feat)  # (B, G, T, N, 2)
        
        # 与基础权重混合
        base = softmax(self.base_weight)
        weights = 0.7 * weights + 0.3 * base
        
        return weights  # [weight_freq, weight_standard]
```

#### 2.2.3 频率一致性作为置信度
```python
# 计算频带间一致性
band_probs = [softmax(logits) for logits in band_logits]
avg_prob = sum(band_probs) / len(band_probs)
js_divergence = mean([kl_div(prob, avg_prob) for prob in band_probs])
consistency = 1.0 / (1.0 + js_divergence)

# 用于调整融合权重
fusion_weights = weight_predictor(features, consistency, freq_weights)
```

---

## 3. 使用方法

### 3.1 配置文件

**V1版本** (原始简化版):
```yaml
USE_FREQ_AWARE: True
USE_FREQ_DECODER_V2: False  # 或不设置
USE_AUX_LOSS: False
```

**V2版本** (改进版):
```yaml
USE_FREQ_AWARE: True
USE_FREQ_DECODER_V2: True   # 使用V2解码器
USE_LEARNABLE_FUSION: True  # 可学习融合
USE_AUX_LOSS: True          # 启用辅助损失
```

### 3.2 训练命令

```bash
# 使用V2版本训练
python train.py --config_path configs/r50_dino_fa_mot_v2_mot17.yaml \
    --outputs_dir ./outputs/fa_mot_v2 \
    --exp_name fa_mot_v2_exp1
```

### 3.3 从V1 checkpoint继续训练V2

```bash
# 先修改配置启用V2，然后resume部分参数
python train.py --config_path configs/r50_dino_fa_mot_v2_mot17.yaml \
    --resume_model ./outputs/v1_checkpoint.pth \
    --resume_optimizer False \
    --resume_scheduler False
```

---

## 4. 实验计划

### 4.1 消融实验

| 实验 | 配置 | 目的 |
|------|------|------|
| Baseline | MOTIP原版 | 基准线 |
| +LFD | V1 w/o FTT | 验证频率分解 |
| +FTT | V1 | 验证时序建模 |
| +V2 Decoder | V2 w/o learnable | 验证完整解码器 |
| +Learnable Fusion | V2 full | 验证可学习融合 |

### 4.2 性能对比表（待填充）

| 方法 | HOTA↑ | IDF1↑ | MOTA↑ | AssA↑ | DetA↑ |
|------|-------|-------|-------|-------|-------|
| MOTIP | - | - | - | - | - |
| FA-MOT V1 | - | - | - | - | - |
| FA-MOT V2 | - | - | - | - | - |

### 4.3 场景分析

| 场景 | 评估指标 | 预期效果 |
|------|---------|---------|
| 严重遮挡 | AssA, IDF1 | 频率分支应该提升 |
| 快速运动 | MOTA | 低频稳定性应该帮助 |
| 密集人群 | HOTA | 一致性约束应该减少ID切换 |

---

## 5. 下一步工作

### 5.1 短期（1-2周）

1. [ ] 运行V2版本训练，对比V1结果
2. [ ] 消融实验验证各组件贡献
3. [ ] 调试形状不匹配的边界情况

### 5.2 中期（2-4周）

1. [ ] DanceTrack数据集测试（遮挡多）
2. [ ] 可视化分析
   - 频率权重分布
   - 融合权重变化
   - 一致性分数与遮挡关系
3. [ ] 超参数搜索
   - num_bands: [3, 4, 5]
   - num_layers: [4, 6, 8]
   - fusion策略

### 5.3 长期（论文准备）

1. [ ] 理论分析与证明
2. [ ] 更多数据集验证（MOT20, SportsMOT）
3. [ ] 与SOTA方法对比
4. [ ] 论文撰写

---

## 6. 文件清单

```
models/motip/
├── freq_aware_id_decoder_v2.py     # 新增: V2版ID解码器
├── freq_aware_trajectory_modeling.py
├── learnable_freq_decomposition.py
├── freq_temporal_transformer.py
├── freq_guided_association.py
├── id_decoder.py                    # 原版ID解码器
├── trajectory_modeling.py           # 原版轨迹建模
└── __init__.py                      # 已更新: 支持V2选择

configs/
├── r50_dino_fa_mot_mot17.yaml       # V1配置
└── r50_dino_fa_mot_v2_mot17.yaml    # V2配置 (新增)
```
