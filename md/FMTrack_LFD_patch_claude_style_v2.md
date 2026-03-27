# LFD (Learnable Frequency Decomposition) 模块修改指南（Claude 风格合并版 v2）

> v2 变更说明（基于 Claude 第二次审阅）：  
> 1) **sigma 范围**：`max_sigma = k/2.0`，避免 envelope 过宽导致滤波器退化为“纯余弦”  
> 2) **use_fixed_laplacian**：只跳过 `filter_loss`，但仍计算 `feature_loss`  
> 3) **overlap_loss（可选）**：只惩罚非对角元素（语义更清晰）

文件：`models/motip/learnable_freq_decomposition.py`

---

## 目录
- [问题 3.1：中心频率参数化不清晰](#问题-31中心频率参数化不清晰frequency-不可解释)
- [问题 3.2：Band importance 粒度太粗](#问题-32band-importance-粒度太粗全-batch-共享权重)
- [问题 3.3：正交性损失作用在输出特征上](#问题-33正交性损失作用在输出特征上容易学成语义拆分)
- [修改检查清单](#修改检查清单)

---

## 问题 3.1：中心频率参数化不清晰（frequency 不可解释）

### 修改位置
**文件**: `models/motip/learnable_freq_decomposition.py`  
**函数**: `LearnableFrequencyFilter._construct_filters`  
**参考行号**: 约 104-133（以你当前版本为准）

### 修改前
```python
def _construct_filters(self) -> torch.Tensor:
    t = torch.linspace(-1, 1, self.kernel_size, device=self.base_filters.device)
    filters = []
    for i in range(self.num_bands):
        base = self.base_filters[i]  # (1, kernel_size)
        center_f = torch.sigmoid(self.center_freqs[i]) * math.pi
        bandwidth = torch.sigmoid(self.bandwidths[i]) * 0.5 + 0.1
        modulation = torch.cos(2 * math.pi * center_f * t)
        envelope = torch.exp(-t**2 / (2 * bandwidth**2))
        final_filter = base * modulation * envelope
        final_filter = final_filter / (final_filter.norm() + 1e-6)
        filters.append(final_filter)
    return torch.stack(filters, dim=0)  # (num_bands, 1, kernel_size)
```

### 修改后（替换整个 `_construct_filters`）
> 核心变化：  
> - 用 cycles/sample 定义中心频率（范围 `[0, 0.5]`，Nyquist）  
> - 用离散采样点 `n = arange(k) - (k-1)/2` 作为时间轴（奇偶 kernel 都居中对齐）  
> - **v2 微调**：sigma 范围为 `[0.8, k/2]`，避免 envelope 过宽导致滤波器“频率选择性很差”

```python
def _construct_filters(self) -> torch.Tensor:
    # - 中心频率: cycles per sample, 范围 [0, 0.5] (Nyquist)
    # - 时间轴: 离散采样点 n, 以样本为单位, 严格居中对齐
    # - 包络: 高斯 envelope, sigma 也是采样点单位
    k = self.kernel_size
    device = self.base_filters.device
    dtype = self.base_filters.dtype

    # 离散时间轴（采样点单位），奇偶 k 都居中对齐
    n = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2.0  # (k,)

    filters = []
    for i in range(self.num_bands):
        base = self.base_filters[i]  # (1, k)

        # 中心频率：固定均匀基频 + 可学习小偏移
        base_freq = (i + 1) / (self.num_bands + 1) * 0.5
        freq_offset = torch.tanh(self.center_freqs[i]) * 0.05
        center_f = torch.clamp(base_freq + freq_offset, 0.02, 0.48)

        # sigma：高斯包络标准差（采样点单位）
        # v2: 避免 sigma 过大使 envelope 近似常数，导致频带高度重叠
        min_sigma = 0.8
        max_sigma = float(k) / 2.0  # e.g. k=7 -> 3.5
        sigma = min_sigma + (max_sigma - min_sigma) * torch.sigmoid(self.bandwidths[i])

        envelope = torch.exp(-0.5 * (n / (sigma + 1e-6)) ** 2)
        modulation = torch.cos(2 * math.pi * center_f * n)

        final_filter = base * modulation * envelope
        final_filter = final_filter / (final_filter.norm() + 1e-6)
        filters.append(final_filter)

    return torch.stack(filters, dim=0)  # (num_bands, 1, k)
```

---

## 问题 3.2：Band importance 粒度太粗（全 batch 共享权重）

### 修改位置
**文件**: `models/motip/learnable_freq_decomposition.py`  
**位置 1**: `__init__` 中 `self.band_importance` 定义（参考行号 72-78）  
**位置 2**: `forward` 中 importance 计算与加权融合（参考行号 174-183）

---

### 3.2.1 修改 `__init__`：band_importance 扩容 + temporal_context

#### 修改前（`__init__`）
```python
self.band_importance = nn.Sequential(
    nn.Linear(dim, dim // 4),
    nn.ReLU(),
    nn.Linear(dim // 4, num_bands),
    nn.Softmax(dim=-1)
)
```

#### 修改后（替换该段 + 新增 temporal_context）
```python
self.band_importance = nn.Sequential(
    nn.Linear(dim, dim // 2),
    nn.GELU(),
    nn.Linear(dim // 2, dim // 4),
    nn.GELU(),
    nn.Linear(dim // 4, num_bands),
    nn.Softmax(dim=-1)
)

self.use_temporal_context = True
if self.use_temporal_context:
    self.temporal_context = nn.Conv1d(
        dim, dim, kernel_size=3, padding=1, groups=dim
    )
```

---

### 3.2.2 修改 `forward`：importance 从 (B,K) → (B,T,N,K)

#### 修改前（`forward`）
```python
x_pool = x.mean(dim=(1, 2))  # (B, C)
importance = self.band_importance(x_pool)  # (B, num_bands)

output = torch.zeros_like(x)
for i, band_feat in enumerate(band_outputs):
    weight = importance[:, i:i+1, None, None]  # (B, 1, 1, 1)
    output = output + weight * band_feat
```

#### 修改后（替换该段）
```python
B, T, N, C = x.shape

if self.use_temporal_context and T > 1:
    x_for_ctx = x.permute(0, 2, 3, 1).reshape(B * N, C, T)
    x_ctx = self.temporal_context(x_for_ctx)
    x_ctx = x_ctx.reshape(B, N, C, T).permute(0, 3, 1, 2)  # (B, T, N, C)
    x_combined = x + x_ctx
else:
    x_combined = x

importance = self.band_importance(x_combined)  # (B, T, N, K)

output = torch.zeros_like(x)
for i, band_feat in enumerate(band_outputs):
    weight = importance[..., i:i+1]  # (B, T, N, 1)
    output = output + weight * band_feat

band_features["importance"] = importance
```

---

## 问题 3.3：正交性损失作用在输出特征上（容易学成“语义拆分”）

### 修改位置
**文件**: `models/motip/learnable_freq_decomposition.py`  
**位置 1**: `LearnableFrequencyFilter.forward` 里 `filters.detach()`  
**位置 2**: `LearnableFrequencyDecomposition.compute_filter_frequency_orthogonality`  
**位置 3**: `LearnableFrequencyDecomposition.compute_orthogonality_loss`

---

### 3.3.1 修复关键点：不要 detach filters（否则频域 orth loss 反传不到滤波器参数）

#### 修改前
```python
band_features['importance'] = importance
band_features['filters'] = filters.detach()
```

#### 修改后（替换为）
```python
band_features['importance'] = importance
band_features['filters'] = filters  # 不要 detach
band_features['filters_detached'] = filters.detach()  # 可选：仅用于日志/可视化
```

---

### 3.3.2 频域 overlap_loss 数值稳定性（v2：只惩罚非对角，语义更清晰）

#### 修改前（示例）
```python
sim = power @ power.transpose(0, 1)  # (K, K)
off = sim - torch.eye(K, device=sim.device, dtype=sim.dtype)
overlap_loss = (off ** 2).mean()
```

#### 修改后（替换该段）
```python
sim = power @ power.transpose(0, 1)  # (K, K)

# v2: 只惩罚非对角元素（频带间重叠）
off_diag_mask = ~torch.eye(K, dtype=torch.bool, device=sim.device)
overlap_loss = sim[off_diag_mask].pow(2).mean()
```

---

### 3.3.3 use_fixed_laplacian 逻辑（v2：只跳过 filter_loss，保留 feature_loss）

#### 修改前（整段早退）
```python
def compute_orthogonality_loss(self, band_features: Dict, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if getattr(self, "use_fixed_laplacian", False):
        for v in band_features.values():
            if isinstance(v, torch.Tensor):
                return v.sum() * 0.0
        return torch.tensor(0.0, device=self.output_norm.weight.device, dtype=self.output_norm.weight.dtype)
    # ...
```

#### 修改后（替换 compute_orthogonality_loss 为最终版）
```python
def compute_orthogonality_loss(self, band_features: Dict, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    # 综合正交性损失：
    # - filter_loss：仅当滤波器可学习时计算
    # - feature_loss：始终计算（即使 fixed filters 也能鼓励 band 输出差异）
    filters = band_features.get("filters", None)

    if getattr(self, "use_fixed_laplacian", False) or filters is None:
        filter_loss = torch.tensor(0.0, device=self.output_norm.weight.device, dtype=self.output_norm.weight.dtype)
    else:
        filter_loss = self.compute_filter_frequency_orthogonality(filters)

    feature_loss = self.compute_feature_orthogonality(band_features, mask)
    return filter_loss + 0.1 * feature_loss
```

---

## 修改检查清单
| 修改点 | 修改文件 | 修改点 | 状态 |
|--------|----------|--------|------|
| 3.1 离散 n 居中对齐 | `learnable_freq_decomposition.py` | `_construct_filters` | ⬜ |
| 3.1 sigma 范围改为 [0.8, k/2] | `learnable_freq_decomposition.py` | `_construct_filters` | ⬜ |
| 3.2 importance 扩容 + temporal_context | `learnable_freq_decomposition.py` | `__init__` | ⬜ |
| 3.2 importance 粒度 (B,T,N,K) | `learnable_freq_decomposition.py` | `forward` | ⬜ |
| 3.3 filters 不 detach | `learnable_freq_decomposition.py` | `forward` | ⬜ |
| 3.3 overlap_loss 只惩罚非对角（可选） | `learnable_freq_decomposition.py` | `compute_filter_frequency_orthogonality` | ⬜ |
| 3.3 fixed_laplacian 保留 feature_loss | `learnable_freq_decomposition.py` | `compute_orthogonality_loss` | ⬜ |

