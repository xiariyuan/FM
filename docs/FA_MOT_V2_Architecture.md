# FA-MOT V2 完整架构详解

> Frequency-Aware Multi-Object Tracking V2
>
> 作者: Research Team
>
> 日期: 2026-01-12

---

## 一、整体数据流

```
输入视频帧 (B, T, 3, H, W)
        ↓
┌───────────────────────────────────────────────────────────────┐
│                    DINO 检测器                                 │
│   ResNet-50 → Deformable Transformer → 检测特征 (B,T,N,256)   │
└───────────────────────────────────────────────────────────────┘
        ↓
┌───────────────────────────────────────────────────────────────┐
│           FrequencyAwareTrajectoryModeling                    │
│   ┌─────────────────────────────────────────────────────┐    │
│   │  1. LearnableFrequencyDecomposition                 │    │
│   │     → 分解为4个频带 [F0, F1, F2, F3]                │    │
│   └─────────────────────────────────────────────────────┘    │
│                          ↓                                    │
│   ┌─────────────────────────────────────────────────────┐    │
│   │  2. FrequencyTemporalTransformer                    │    │
│   │     → 分频带时序建模 + 跨频带交互                    │    │
│   └─────────────────────────────────────────────────────┘    │
│                          ↓                                    │
│   输出: enhanced_features + freq_band_features               │
└───────────────────────────────────────────────────────────────┘
        ↓
┌───────────────────────────────────────────────────────────────┐
│              FrequencyAwareIDDecoderV3 (双分支)               │
│                                                               │
│   ┌──────────────────┐      ┌──────────────────┐             │
│   │   标准分支        │      │   频率分支        │             │
│   │  Mamba+CrossAttn │      │  多频带ID预测     │             │
│   │  → std_logits    │      │  → freq_logits   │             │
│   └────────┬─────────┘      └────────┬─────────┘             │
│            └──────────┬──────────────┘                        │
│                       ↓                                       │
│            ┌──────────────────┐                               │
│            │  AttentionFusion │                               │
│            │  → fused_logits  │                               │
│            └──────────────────┘                               │
└───────────────────────────────────────────────────────────────┘
        ↓
    ID预测 (B, G, T, N, 51)
```

---

## 二、模块详解

### 1. LearnableFrequencyDecomposition (可学习频率分解)

**文件**: `models/motip/learnable_freq_decomposition.py:198-369`

**核心思想**: 将轨迹特征分解为4个可学习的频率带，不同频带捕获不同时间尺度的运动模式。

#### 1.1 LearnableFrequencyFilter (核心滤波器)

```python
# 滤波器构造 (Gabor-like)
t = linspace(-1, 1, kernel_size=7)  # 时间轴

for i in range(num_bands=4):
    # 中心频率 (可学习)
    center_f = sigmoid(center_freqs[i]) * π  # [0, π]

    # 带宽 (可学习)
    bandwidth = sigmoid(bandwidths[i]) * 0.5 + 0.1  # [0.1, 0.6]

    # Gabor调制
    modulation = cos(2π * center_f * t)  # 频率调制
    envelope = exp(-t² / (2 * bandwidth²))  # 高斯包络

    # 最终滤波器 = 基础滤波器 × 调制 × 包络
    filter[i] = base_filters[i] * modulation * envelope
```

**初始化**: 使用 DOG (Difference of Gaussians) 初始化，提供良好的起点：
- Band 0: σ=0.5 (高频)
- Band 1: σ=0.8 (中高频)
- Band 2: σ=1.1 (中低频)
- Band 3: σ=1.4 (低频)

#### 1.2 频带重要性预测

```python
# 根据输入内容自适应调整频带权重
x_pool = x.mean(dim=(1, 2))  # 全局池化 (B, C)
importance = band_importance(x_pool)  # (B, 4) Softmax归一化

# 加权融合
output = Σ importance[i] * band_features[i]
```

#### 1.3 正交性损失

```python
def compute_orthogonality_loss(band_features):
    """确保不同频带捕获不同信息"""
    # all_bands: (B*T*N, C, K=4)
    bands_norm = F.normalize(bands_flat, dim=1)  # L2归一化

    # 计算频带相关矩阵 (K × K)
    correlation = bands_norm.T @ bands_norm

    # 正交性损失: 非对角元素应接近0
    identity = eye(K)
    ortho_loss = ||correlation - identity||²

    return ortho_loss  # 权重: 0.1
```

---

### 2. FrequencyTemporalTransformer (频率-时序Transformer)

**文件**: `models/motip/freq_temporal_transformer.py:372-564`

**核心思想**: 不同频率使用不同的时序建模范围 + 跨频带信息交互

#### 2.1 FrequencyAwarePositionalEncoding

```python
# 不同频带使用不同尺度的位置编码
scale = sigmoid(scale_factors[band_idx]) * 2 + 0.5  # [0.5, 2.5]

# 低频带(band_idx=0): scale大 → 位置编码变化慢 → 捕获长期依赖
# 高频带(band_idx=3): scale小 → 位置编码变化快 → 捕获短期细节

pe = interpolate(base_pe, positions / scale) + band_offsets[band_idx]
```

#### 2.2 BandSpecificTemporalAttention

```python
# 根据频带类型调整注意力窗口
window_ratio = 1.0 - (band_idx / 3) * 0.7  # [0.3, 1.0]
window_size = max(3, int(30 * window_ratio))

# Band 0 (低频): window=30 (全局注意力)
# Band 1: window=23
# Band 2: window=16
# Band 3 (高频): window=9 (局部注意力)

# 相对位置偏置
rel_pos_bias = Parameter(2*max_len-1, num_heads)
attn = Q @ K.T * scale + rel_pos_bias[relative_positions]
```

#### 2.3 CrossBandInteraction (跨频带交互)

```python
# 让不同频带交换信息
# 1. 添加频带嵌入
for i, feat in enumerate(band_features):
    feat = feat + band_embeddings[i]  # 区分不同频带

# 2. 跨频带自注意力
stacked = stack(band_features, dim=3)  # (B, T, N, K, C)
attended = cross_attn(stacked, stacked, stacked)

# 3. 门控残差
gate = sigmoid(Linear(concat(original, attended)))
output = original + gate * (attended - original)
```

---

### 3. FrequencyGuidedAssociation (频率引导关联)

**文件**: `models/motip/freq_guided_association.py:320-499`

**核心思想**: 利用频率信息指导目标关联，特别是在遮挡场景下

#### 3.1 OcclusionEstimator (遮挡估计器)

```python
def estimate_occlusion(features, band_energies):
    """基于时序变化和频率分布估计遮挡程度"""

    # 1. 时序差分编码
    temporal_diff = features[:, 1:] - features[:, :-1]
    temporal_encoding = temporal_diff_encoder(temporal_diff)  # (B,T,N,64)

    # 2. 频率分布编码
    freq_encoding = freq_dist_encoder(band_energies)  # (B,T,N,64)

    # 3. 预测遮挡概率
    combined = concat(temporal_encoding, freq_encoding)
    occlusion_scores = sigmoid(occlusion_head(combined))  # [0, 1]

    return occlusion_scores
```

**遮挡检测原理**:
- 时序差分大 → 可能遮挡（特征突变）
- 高频能量突然下降 → 可能遮挡（细节丢失）

#### 3.2 FrequencyWeightPredictor (频率权重预测)

```python
def predict_freq_weights(features, occlusion_scores):
    """根据遮挡程度动态调整频率权重"""

    # 内容编码
    content = content_encoder(features)  # (B,T,N,64)

    # 遮挡条件编码
    occlusion = occlusion_encoder(occlusion_scores)  # (B,T,N,64)

    # 预测权重偏移
    weight_logits = weight_predictor(concat(content, occlusion))

    # 结合先验权重
    # prior_weights = [0.4, 0.3, 0.2, 0.1] (高频优先)
    weights = softmax(prior_weights + modulation * weight_logits)

    return weights  # (B, T, N, 4)
```

**频率权重调整策略**:
```
正常场景 (occlusion < 0.3):
  weights ≈ [0.4, 0.3, 0.2, 0.1]  # 高频优先，精确匹配

遮挡场景 (occlusion > 0.7):
  weights ≈ [0.1, 0.2, 0.3, 0.4]  # 低频优先，稳定关联
```

#### 3.3 MultiBandIDDecoder (多频带ID解码)

```python
def decode_id(band_id_embeds, freq_weights):
    """各频带独立预测ID，然后加权融合"""

    # 1. 各频带独立预测
    band_logits = []
    for i, embed in enumerate(band_id_embeds):
        logits = band_embed_to_word[i](embed)  # (B,T,N,51)
        band_logits.append(logits)

    # 2. 加权融合
    stacked = stack(band_logits, dim=-1)  # (B,T,N,51,4)
    weighted = (stacked * freq_weights.unsqueeze(-2)).sum(dim=-1)

    # 3. MLP融合（学习非线性组合）
    concat_logits = stacked.reshape(..., 51*4)
    mlp_fused = band_fusion(concat_logits)

    # 4. 组合两种融合
    final_logits = 0.5 * weighted + 0.5 * mlp_fused

    # 5. 一致性评估
    consistency = consistency_evaluator(concat_logits)  # [0, 1]

    return final_logits, consistency
```

---

### 4. FrequencyAwareIDDecoderV3 (双分支ID解码器)

**文件**: `models/motip/freq_aware_id_decoder_v2.py:373-814`

**核心思想**: 双分支独立监督 + 注意力融合

#### 4.1 标准分支 (Standard Branch)

```python
# 6层 Mamba + CrossAttention 解码器
for layer in range(6):
    # 1. Mamba自注意力 (层1-5)
    if layer > 0:
        unknown_embeds = mamba_layers[layer-1](unknown_embeds)

    # 2. 交叉注意力 (unknown → trajectory)
    cross_out = cross_attn(
        query=unknown_embeds,
        key=trajectory_embeds,
        value=trajectory_embeds,
        attn_mask=cross_attn_mask + rel_pos_bias
    )

    # 3. FFN
    unknown_embeds = ffn(cross_out)

    # 4. 每层预测ID (辅助损失)
    layer_logits = embed_to_word(unknown_embeds[..., -id_dim:])
    std_layer_logits.append(layer_logits)

std_logits = std_layer_logits[-1]  # 最终输出
```

#### 4.2 频率分支 (Frequency Branch)

```python
class FrequencyBranch:
    def forward(unknown_band_features):
        """基于多频带特征的ID预测"""

        band_logits_list = []
        band_confidence_list = []

        for i in range(4):  # 4个频带
            # 1. 特征增强
            enhanced = band_enhancers[i](unknown_band_features[i])

            # 2. ID预测
            logits = band_id_heads[i](enhanced)  # (B,G,T,N,51)
            band_logits_list.append(logits)

            # 3. 置信度预测
            confidence = band_confidence[i](enhanced)  # (B,G,T,N,1)
            band_confidence_list.append(confidence)

        # 4. 基于置信度的加权融合
        conf_weights = softmax(stack(band_confidence_list))
        weighted_logits = (stack(band_logits_list) * conf_weights).sum()

        # 5. MLP融合
        mlp_fused = band_fusion(concat(band_logits_list))

        # 6. 组合 (可学习权重)
        alpha = sigmoid(fusion_alpha)  # 初始0.5
        freq_logits = alpha * weighted_logits + (1-alpha) * mlp_fused

        return freq_logits, freq_confidence
```

#### 4.3 AttentionFusion (注意力融合)

```python
class AttentionFusion:
    def forward(freq_logits, std_logits, features, freq_confidence, band_confidence):
        """自适应融合双分支预测"""

        # 1. 计算标准分支置信度 (基于熵)
        std_probs = softmax(std_logits)
        std_entropy = -(std_probs * log(std_probs)).sum(dim=-1)
        std_confidence = 1 / (1 + std_entropy)  # 熵越低，置信度越高

        # 2. 评估两分支一致性
        consistency_input = concat(freq_logits, std_logits)
        consistency = consistency_net(consistency_input)  # [0, 1]

        # 3. 预测融合权重
        fusion_input = concat(
            freq_logits, std_logits, features,
            band_confidence, freq_confidence, std_confidence
        )
        raw_weights = fusion_net(fusion_input)  # (B,G,T,N,2)

        # 4. 结合先验权重 [0.5, 0.5]
        prior = softmax(prior_weights)
        fusion_weights = softmax(raw_weights) * 0.7 + prior * 0.3

        # 5. 加权融合
        fused = fusion_weights[...,0:1] * freq_logits +
                fusion_weights[...,1:2] * std_logits

        # 6. 一致性门控残差
        gate = consistency * residual_gate(fused)

        # 高一致性 → 信任融合结果
        # 低一致性 → 回退到标准分支
        final_logits = gate * fused + (1 - gate) * std_logits

        return final_logits
```

---

### 5. 损失函数设计

```python
Total_Loss = L_detr + L_id

L_id = L_std + λ1*L_freq + λ2*L_fusion + λ3*L_consist + λ4*L_ortho

其中:
- L_std: 标准分支各层的交叉熵损失 (辅助损失)
- L_freq: 频率分支的交叉熵损失 (权重: 1.0)
- L_fusion: 融合结果的交叉熵损失 (权重: 1.0)
- L_consist: 双分支一致性损失 (JS散度, 权重: 0.05)
- L_ortho: 频带正交性损失 (权重: 0.1)
```

**一致性损失计算**:
```python
def compute_consistency_loss(freq_logits, std_logits):
    """JS散度衡量双分支预测一致性"""
    freq_p = softmax(freq_logits / temperature)
    std_p = softmax(std_logits / temperature)

    avg_p = 0.5 * (freq_p + std_p)

    kl_freq = KL(freq_p || avg_p)
    kl_std = KL(std_p || avg_p)

    js_divergence = 0.5 * (kl_freq + kl_std)
    return js_divergence
```

---

## 三、频域引导的核心机制总结

### 引导方式1: 频率分解 → 多尺度运动捕获

```
输入特征 → 4个频带滤波器 → 4个频带特征

Band 0 (高频): 捕获快速运动、细节变化
Band 1 (中高频): 捕获正常运动
Band 2 (中低频): 捕获缓慢运动
Band 3 (低频): 捕获长期趋势、对遮挡鲁棒
```

### 引导方式2: 遮挡感知 → 动态频率权重

```
正常场景:
  高频权重大 → 精确匹配 → 高AssA

遮挡场景:
  低频权重大 → 稳定关联 → 减少ID Switch
```

### 引导方式3: 多频带投票 → 鲁棒ID预测

```
4个频带独立预测ID → 置信度加权 → 融合预测

当某个频带受干扰时，其他频带可以补偿
```

### 引导方式4: 双分支融合 → 互补增强

```
标准分支: 基于时序注意力的精确匹配
频率分支: 基于频率特征的鲁棒预测

一致性高 → 信任融合结果
一致性低 → 回退到标准分支 (保守策略)
```

---

## 四、关键参数配置

| 参数 | 值 | 说明 |
|------|-----|------|
| num_freq_bands | 4 | 频带数量 |
| freq_kernel_size | 7 | 滤波器核大小 |
| num_freq_temporal_layers | 2 | 时序Transformer层数 |
| freq_temporal_heads | 8 | 注意力头数 |
| freq_ortho_loss_weight | 0.1 | 正交性损失权重 |
| freq_consistency_loss_weight | 0.05 | 一致性损失权重 |
| use_mamba_for_lowfreq | True | 低频带使用Mamba |
| use_occlusion_aware | True | 启用遮挡感知 |
| num_id_decoder_layers | 6 | ID解码器层数 |
| rel_pe_length | 30 | 相对位置编码长度 |
| id_dim | 256 | ID嵌入维度 |
| feature_dim | 256 | 特征维度 |

---

## 五、文件结构

```
models/motip/
├── learnable_freq_decomposition.py   # 可学习频率分解
│   ├── LearnableFrequencyFilter      # Gabor-like滤波器
│   └── LearnableFrequencyDecomposition # 完整分解模块
│
├── freq_temporal_transformer.py      # 频率-时序Transformer
│   ├── FrequencyAwarePositionalEncoding # 频率感知位置编码
│   ├── BandSpecificTemporalAttention    # 频带特定时序注意力
│   ├── CrossBandInteraction             # 跨频带交互
│   └── FrequencyTemporalTransformer     # 完整FTT模块
│
├── freq_guided_association.py        # 频率引导关联
│   ├── OcclusionEstimator            # 遮挡估计器
│   ├── FrequencyWeightPredictor      # 频率权重预测
│   ├── MultiBandIDDecoder            # 多频带ID解码
│   └── FrequencyGuidedAssociation    # 完整FGA模块
│
├── freq_aware_trajectory_modeling.py # 轨迹建模整合
│   └── FrequencyAwareTrajectoryModeling # 整合LFD+FTT
│
└── freq_aware_id_decoder_v2.py       # 双分支ID解码器
    ├── FrequencyBranch               # 频率分支
    ├── AttentionFusion               # 注意力融合
    └── FrequencyAwareIDDecoderV3     # 完整解码器
```

---

## 六、训练配置示例

```yaml
# configs/r50_dino_fa_mot_v2_mot17.yaml

# 频率感知模块
use_freq_aware: true
use_freq_decoder_v2: true
use_learnable_fusion: true
num_freq_bands: 4
freq_kernel_size: 7
use_fixed_laplacian: false
num_freq_temporal_layers: 2
freq_temporal_heads: 8
use_mamba_for_lowfreq: true
freq_dropout: 0.1

# 频率引导关联
use_freq_guided_assoc: true
use_occlusion_aware: true

# 损失权重
freq_ortho_loss_weight: 0.1
freq_consistency_loss_weight: 0.05

# 训练参数
epochs: 12
batch_size: 1
lr: 0.0001
scheduler_milestones: [8, 10]
```

---

## 七、推理流程

```python
def inference(video_frames):
    # 1. DINO检测
    detections = dino_detector(video_frames)

    # 2. 构建轨迹
    trajectories = build_trajectories(detections)

    # 3. 频率感知轨迹建模
    seq_info = trajectory_modeling(trajectories)
    # → 输出: enhanced_features + freq_band_features

    # 4. 双分支ID解码
    fused_logits, _, _, extra_info = id_decoder(seq_info)

    # 5. ID分配
    id_predictions = fused_logits.argmax(dim=-1)

    # 6. 轨迹关联
    tracks = associate_tracks(trajectories, id_predictions)

    return tracks
```

---

## 八、频率分解原理详解

### 8.1 核心原理：时域滤波 → 频率响应

频率分解是在**时间维度T**上进行的，不是空间维度。

```
输入: trajectory_features (B, G, T=30, N, C=256)
                              ↑
                         时间维度（30帧）
```

### 8.2 滤波器如何区分高低频

```python
# learnable_freq_decomposition.py:86-102

def _init_filters(self):
    """DOG (Difference of Gaussians) 初始化"""
    for i in range(self.num_bands):  # i = 0, 1, 2, 3
        t = torch.linspace(-3, 3, self.kernel_size)  # kernel_size=7

        # 关键：不同band使用不同的sigma
        sigma1 = 0.5 + i * 0.3  # Band0=0.5, Band1=0.8, Band2=1.1, Band3=1.4
        sigma2 = sigma1 * 1.6

        g1 = torch.exp(-t**2 / (2 * sigma1**2))  # 窄高斯
        g2 = torch.exp(-t**2 / (2 * sigma2**2))  # 宽高斯
        dog = g1 - g2  # 高斯差分 = 带通滤波器
```

**信号处理原理**：
- **小sigma (0.5)** → 窄的时域响应 → 宽的频域响应 → **高频滤波器**
- **大sigma (1.4)** → 宽的时域响应 → 窄的频域响应 → **低频滤波器**

### 8.3 可视化滤波器响应

```
Band 0 (sigma=0.5) - 高频滤波器:
时域:    ___/‾‾‾\___     (窄峰，快速响应)
频域:    ‾‾‾‾‾‾‾‾‾‾‾     (宽带，响应高频变化)

Band 3 (sigma=1.4) - 低频滤波器:
时域:    __/‾‾‾‾‾‾‾\__   (宽峰，平滑响应)
频域:    ___/‾‾‾\___     (窄带，只响应低频变化)
```

### 8.4 时序卷积过程

```python
# learnable_freq_decomposition.py:155-172

def forward(self, x):
    B, T, N, C = x.shape

    # 重排为 (B*N, C, T) 用于时间维度卷积
    x_conv = x.permute(0, 2, 3, 1).reshape(B * N, C, T)

    for i in range(self.num_bands):
        # 在时间维度T上进行卷积
        filtered = F.conv1d(x_conv, filter[i], padding=3, groups=C)
        #                          ↑
        #                    kernel_size=7, 在T维度滑动
```

### 8.5 不同频带捕获不同运动的原理

假设一个目标的特征在30帧中的变化：

```
帧:     1  2  3  4  5  6  7  8  9  10 ... 30
        ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓

快速运动目标 (特征剧烈变化):
特征值: ↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        高频滤波器(Band 0)强响应 ✓

缓慢运动目标 (特征平缓变化):
特征值: ─────────/‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
                 ^^^^^^^^^^^^^^^^^^^^
        低频滤波器(Band 3)强响应 ✓

遮挡目标 (特征突然消失后恢复):
特征值: ‾‾‾‾‾‾‾‾‾‾___________‾‾‾‾‾‾‾‾‾
        ^^^^^^^^^^           ^^^^^^^^^^
        低频保持整体趋势，高频丢失细节
        → 低频更鲁棒 ✓
```

### 8.6 具体数学解释

```python
# 高频滤波器 (Band 0, sigma=0.5)
# 对相邻帧差异敏感

输入序列: [f1, f2, f3, f4, f5, f6, f7]  # 7帧特征
高频核:   [0.1, -0.3, 0.5, -0.6, 0.5, -0.3, 0.1]  # 类似边缘检测

输出 ≈ 0.1*f1 - 0.3*f2 + 0.5*f3 - 0.6*f4 + 0.5*f5 - 0.3*f6 + 0.1*f7
     ≈ 帧间差异的加权和
     → 快速变化时输出大，缓慢变化时输出小


# 低频滤波器 (Band 3, sigma=1.4)
# 对整体趋势敏感

低频核:   [0.1, 0.15, 0.2, 0.3, 0.2, 0.15, 0.1]  # 类似平滑

输出 ≈ 0.1*f1 + 0.15*f2 + 0.2*f3 + 0.3*f4 + 0.2*f5 + 0.15*f6 + 0.1*f7
     ≈ 加权平均
     → 捕获整体趋势，忽略快速波动
```

### 8.7 Gabor调制进一步精确控制

```python
# learnable_freq_decomposition.py:104-133

def _construct_filters(self):
    for i in range(self.num_bands):
        # 中心频率 (可学习)
        center_f = sigmoid(center_freqs[i]) * π

        # Gabor调制 = 余弦 × 高斯包络
        modulation = cos(2π * center_f * t)  # 控制响应的中心频率
        envelope = exp(-t² / (2 * bandwidth²))  # 控制频带宽度

        final_filter = base_filter * modulation * envelope
```

**Gabor滤波器的优势**：
- 在时域和频域都有良好的局部化
- 可以精确控制中心频率和带宽
- 类似人类视觉系统的频率选择性

### 8.8 频带与运动模式对应关系

| 场景 | 特征变化 | 响应频带 | 效果 |
|------|----------|----------|------|
| 快速运动 | 帧间差异大 | Band 0 (高频) | 精确捕获运动细节 |
| 正常运动 | 中等变化 | Band 1-2 | 平衡精度和稳定性 |
| 缓慢运动 | 帧间差异小 | Band 3 (低频) | 捕获长期趋势 |
| 遮挡 | 高频信息丢失 | Band 3 (低频) | 保持整体轨迹，鲁棒关联 |

**核心思想**：不同速度的运动在时间序列上表现为不同频率的变化，通过多频带分解可以分别捕获这些不同尺度的运动模式。

---

*文档生成时间: 2026-01-12*
