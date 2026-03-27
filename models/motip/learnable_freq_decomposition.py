# Copyright (c) 2024. All Rights Reserved.
"""
Learnable Frequency Decomposition (LFD) Module

核心创新点:
1. 可学习的频率滤波器组（替代固定拉普拉斯）
2. 自适应频带分配（根据输入动态调整）
3. 频率正交性约束（确保不同频带捕获不同信息）

理论基础:
- 传统拉普拉斯金字塔使用固定滤波器，无法适应不同场景
- 我们提出可学习的频率分解，让网络自己学习最优的频率划分
- 同时引入正交性约束，避免频带之间的信息冗余
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict


class LearnableFrequencyFilter(nn.Module):
    """
    可学习的频率滤波器
    
    不同于固定的拉普拉斯核，我们学习一组滤波器，每个滤波器
    专注于捕获特定的频率成分。滤波器的响应在频率域上是可解释的。
    """
    
    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        num_bands: int = 4,
        use_fixed_filters: bool = False,
        use_temporal_context: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.num_bands = num_bands
        self.use_fixed_filters = use_fixed_filters
        
        # 可学习的滤波器参数
        # 使用分解形式：每个滤波器 = base_filter * frequency_modulation
        self.base_filters = nn.Parameter(
            torch.randn(num_bands, 1, kernel_size) * 0.02
        )
        
        # 频率调制参数（控制每个滤波器的中心频率和带宽）
        # center_freq: [0, 1] 表示从低频到高频
        # bandwidth: 控制频带宽度
        # 单调中心频率参数化：raw delta 参数（softplus 保证 > 0）
        # 通过 cumsum(softplus(delta)) 构造单调递增的中心频率，避免频带交换/坍塌
        self.freq_deltas = nn.Parameter(
            torch.ones(num_bands) * 0.5
        )
        self.bandwidths = nn.Parameter(
            torch.ones(num_bands) * 0.2
        )

        # Cached diagnostics (do not mutate module attributes in forward; safe for DDP/torch.compile).
        self.register_buffer("_cached_center_freqs", torch.zeros(num_bands, dtype=torch.float32))
        self.register_buffer("_cached_sigmas", torch.zeros(num_bands, dtype=torch.float32))
        
        # 输出投影
        self.out_projs = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_bands)
        ])
        
        # 频带重要性（可学习的软注意力）
        # 注意：这里只输出 logits，softmax( / tau ) 在 forward 中进行（便于 train.py 统一控制 tau，保证可复现）
        self.band_importance = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, num_bands),  # logits
        )
        # temperature for importance softmax (set in train.py per-epoch / per-step)
        self.register_buffer("importance_tau", torch.tensor(1.0))
        self.use_temporal_context = bool(use_temporal_context)
        if self.use_temporal_context:
            self.temporal_context = nn.Conv1d(
                dim, dim, kernel_size=3, padding=1, groups=dim
            )
        
        self._init_filters()
        if self.use_fixed_filters:
            self.base_filters.requires_grad_(False)
            self.freq_deltas.requires_grad_(False)
            self.bandwidths.requires_grad_(False)
    
    def _init_filters(self):
        """
        初始化滤波器为类似DOG(Difference of Gaussians)的形式
        这提供了一个好的起点，网络可以从这里学习
        """
        with torch.no_grad():
            for i in range(self.num_bands):
                # 创建带通滤波器的初始化
                t = torch.linspace(-3, 3, self.kernel_size)
                # 高斯差分，不同band有不同的sigma
                sigma1 = 0.5 + i * 0.3
                sigma2 = sigma1 * 1.6
                g1 = torch.exp(-t**2 / (2 * sigma1**2))
                g2 = torch.exp(-t**2 / (2 * sigma2**2))
                dog = g1 - g2
                dog = dog / (dog.abs().sum() + 1e-6)  # 归一化
                self.base_filters.data[i, 0] = dog
    
    def _construct_filters(self) -> torch.Tensor:
        """
        构造最终的滤波器组
        
        使用频率调制来确保滤波器在频率域上有明确的响应
        """
        # - 中心频率: cycles per sample, 范围 [0, 0.5] (Nyquist)
        # - 时间轴: 离散采样点 n, 以样本为单位, 严格居中对齐
        # - 包络: 高斯 envelope, sigma 也是采样点单位
        k = self.kernel_size
        device = self.base_filters.device
        dtype = self.base_filters.dtype

        # 离散时间轴（采样点单位），奇偶 k 都居中对齐
        n = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2.0  # (k,)

        # ========== 单调中心频率参数化（cycles/sample） ==========
        # deltas > 0 -> cumsum -> strictly increasing
        deltas = F.softplus(self.freq_deltas) + 1e-4  # (K,)
        raw_freqs = torch.cumsum(deltas, dim=0)       # (K,)
        raw_freqs = raw_freqs / (raw_freqs[-1] + 1e-6)
        min_freq, max_freq = 0.02, 0.48              # avoid DC & Nyquist boundary
        center_freqs = min_freq + raw_freqs * (max_freq - min_freq)  # (K,)
        # ==========================================================

        filters = []
        sigmas = []
        for i in range(self.num_bands):
            # 基础滤波器
            base = self.base_filters[i]  # (1, k)

            # 中心频率（已确保单调递增）：cycles per sample
            center_f = center_freqs[i]

            # sigma：高斯包络标准差（采样点单位）
            # v2: 避免 sigma 过大使 envelope 近似常数，导致频带高度重叠
            min_sigma = 0.8
            max_sigma = float(k) / 2.0  # e.g. k=7 -> 3.5
            sigma = min_sigma + (max_sigma - min_sigma) * torch.sigmoid(self.bandwidths[i])

            sigmas.append(sigma)

            envelope = torch.exp(-0.5 * (n / (sigma + 1e-6)) ** 2)
            modulation = torch.cos(2 * math.pi * center_f * n)

            # 最终滤波器
            final_filter = base * modulation * envelope

            # DC suppression：对带通/高通 band 做零均值，抑制低频/DC 偏置（建议只对 i>0）
            if i > 0:
                final_filter = final_filter - final_filter.mean(dim=-1, keepdim=True)

            # 归一化（保持能量）
            final_filter = final_filter / final_filter.norm().clamp(min=1e-6)
            filters.append(final_filter)

        # cache for diagnostics/logging (no grad)
        with torch.no_grad():
            if self._cached_center_freqs.numel() == center_freqs.numel():
                self._cached_center_freqs.copy_(center_freqs.detach().to(self._cached_center_freqs.dtype))
            if len(sigmas) > 0:
                _sigmas = torch.stack(sigmas, dim=0).detach()
                if self._cached_sigmas.numel() == _sigmas.numel():
                    self._cached_sigmas.copy_(_sigmas.to(self._cached_sigmas.dtype))

        return torch.stack(filters, dim=0)  # (num_bands, 1, k)
    
    def forward(
        self, 
        x: torch.Tensor,
        return_all_bands: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: (B, T, N, C) 时序特征
            return_all_bands: 是否返回所有频带
            
        Returns:
            output: 融合后的特征 (B, T, N, C)
            band_features: 各频带特征字典
        """
        B, T, N, C = x.shape
        
        # 构造滤波器
        filters = self._construct_filters()  # (num_bands, 1, kernel_size)
        
        # 重排用于卷积: (B*N, C, T)
        x_conv = x.permute(0, 2, 3, 1).reshape(B * N, C, T)
        
        # 对每个频带应用滤波
        band_outputs = []
        for i in range(self.num_bands):
            # 扩展滤波器到所有通道
            f = filters[i].expand(C, 1, -1)  # (C, 1, kernel_size)
            
            # 分组卷积（每个通道独立滤波）
            padding = self.kernel_size // 2
            filtered = F.conv1d(x_conv, f, padding=padding, groups=C)
            
            # 重排回原始形状: (B, T, N, C)
            filtered = filtered.reshape(B, N, C, T).permute(0, 3, 1, 2)
            
            # 通过投影层
            filtered = self.out_projs[i](filtered)
            band_outputs.append(filtered)
        
        # 计算频带重要性（基于输入内容自适应）
        if self.use_temporal_context and T > 1:
            x_for_ctx = x.permute(0, 2, 3, 1).reshape(B * N, C, T)
            x_ctx = self.temporal_context(x_for_ctx)
            x_ctx = x_ctx.reshape(B, N, C, T).permute(0, 3, 1, 2)  # (B, T, N, C)
            x_combined = x + x_ctx
        else:
            x_combined = x

        importance_logits = self.band_importance(x_combined)  # (B, T, N, num_bands)
        tau = float(self.importance_tau.item()) if hasattr(self, 'importance_tau') else 1.0
        importance = F.softmax(importance_logits / max(tau, 1e-6), dim=-1)

        # 加权融合
        output = torch.zeros_like(x)
        for i, band_feat in enumerate(band_outputs):
            weight = importance[..., i:i+1]  # (B, T, N, 1)
            output = output + weight * band_feat
        
        # 准备返回的频带特征字典
        band_features = {
            f'band_{i}': band_outputs[i] for i in range(self.num_bands)
        }
        band_features['importance'] = importance
        band_features['importance_logits'] = importance_logits
        band_features['importance_tau'] = torch.tensor(tau, device=importance.device, dtype=importance.dtype)
        if hasattr(self, '_cached_center_freqs') and self._cached_center_freqs is not None:
            band_features['center_freqs'] = self._cached_center_freqs.to(device=importance.device)
        if hasattr(self, '_cached_sigmas') and self._cached_sigmas is not None:
            band_features['sigmas'] = self._cached_sigmas.to(device=importance.device)
        band_features['filters'] = filters  # 不要 detach
        band_features['filters_detached'] = filters.detach()
        
        if return_all_bands:
            # 也返回拼接的多频带特征，供后续模块使用
            band_features['all_bands'] = torch.stack(band_outputs, dim=-1)  # (B, T, N, C, num_bands)
        
        return output, band_features


class LearnableFrequencyDecomposition(nn.Module):
    """
    完整的可学习频率分解模块
    
    包含:
    1. 可学习频率滤波器
    2. 频率正交性约束（损失函数）
    3. 残差连接和归一化
    """
    
    def __init__(
        self,
        dim: int,
        num_bands: int = 4,
        kernel_size: int = 7,
        dropout: float = 0.1,
        freq_ortho_metric: str = "dot",
        use_fixed_laplacian: bool = False,
        feature_ortho_weight: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_bands = num_bands
        self.use_fixed_laplacian = use_fixed_laplacian

        self.freq_ortho_metric = freq_ortho_metric
        self.feature_ortho_weight = float(feature_ortho_weight)
        
        # 输入归一化
        self.input_norm = nn.LayerNorm(dim)
        
        # 可学习频率滤波器
        self.freq_filter = LearnableFrequencyFilter(
            dim=dim,
            kernel_size=kernel_size,
            num_bands=num_bands,
            use_fixed_filters=use_fixed_laplacian,
        )
        
        # 频带融合网络
        self.band_fusion = nn.Sequential(
            nn.Linear(dim * num_bands, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        
        # 输出门控
        self.output_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        
        # 输出归一化
        self.output_norm = nn.LayerNorm(dim)
        
    def compute_filter_frequency_orthogonality(self, filters: torch.Tensor) -> torch.Tensor:
        """
        频域重叠损失：鼓励不同频带的频谱能量分布相互独立
        """
        if filters is None or filters.numel() == 0:
            return torch.tensor(0.0, device=self.output_norm.weight.device, dtype=self.output_norm.weight.dtype)

        filt = filters
        if filt.dim() == 3:
            filt = filt.squeeze(1)
        elif filt.dim() > 3:
            filt = filt.reshape(filt.shape[0], -1)

        K = filt.shape[0]
        if K <= 1:
            return torch.tensor(0.0, device=filt.device, dtype=filt.dtype)

        n_fft = max(256, int(filt.shape[-1]) * 8)
        fft = torch.fft.rfft(filt, n=n_fft, dim=-1)
        power_raw = fft.real.pow(2) + fft.imag.pow(2)  # (K, F)

        metric = getattr(self, "freq_ortho_metric", "dot")
        metric = str(metric).lower()

        if metric in ("dot", "cos", "cosine"):
            power = power_raw / (power_raw.norm(dim=-1, keepdim=True) + 1e-6)
            sim = power @ power.transpose(0, 1)  # (K, K)
            off_diag_mask = ~torch.eye(K, dtype=torch.bool, device=sim.device)
            if not off_diag_mask.any():
                return torch.tensor(0.0, device=sim.device, dtype=sim.dtype)
            overlap_loss = sim[off_diag_mask].pow(2).mean()
            return overlap_loss

        if metric in ("js", "jensen-shannon", "jsd"):
            p = power_raw / (power_raw.sum(dim=-1, keepdim=True) + 1e-8)
            eps = 1e-8
            js_vals = []
            for i in range(K):
                for j in range(i + 1, K):
                    pi = p[i].clamp(min=eps)
                    pj = p[j].clamp(min=eps)
                    m = 0.5 * (pi + pj)
                    kl_i = (pi * (pi.log() - m.log())).sum()
                    kl_j = (pj * (pj.log() - m.log())).sum()
                    js = 0.5 * (kl_i + kl_j)
                    js_vals.append(js)
            if len(js_vals) == 0:
                return torch.tensor(0.0, device=filt.device, dtype=filt.dtype)
            js_mean = torch.stack(js_vals).mean()
            js_max = math.log(2.0)
            overlap_loss = 1.0 - (js_mean / js_max).clamp(0.0, 1.0)
            return overlap_loss

        raise ValueError(f"Unknown freq_ortho_metric: {metric}")

    def compute_feature_orthogonality(
        self,
        band_features: Dict,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        特征正交性损失：鼓励各频带输出特征互相区分
        """
        if 'all_bands' not in band_features:
            # 使用band_features中的tensor获取设备，避免next(self.parameters())失败
            for v in band_features.values():
                if isinstance(v, torch.Tensor):
                    return torch.tensor(0.0, device=v.device, dtype=v.dtype)
            return torch.tensor(0.0, device=self.output_norm.weight.device, dtype=self.output_norm.weight.dtype)

        all_bands = band_features['all_bands']  # (B, T, N, C, num_bands)
        _, _, N, C, K = all_bands.shape

        # 边界检查：空目标
        if N == 0:
            return torch.tensor(0.0, device=all_bands.device, dtype=all_bands.dtype)

        # 重排为 (B*T*N, C, K)
        bands_flat = all_bands.reshape(-1, C, K)

        if mask is not None:
            # mask 形状: (B, G, T, N) 或 (B, T, N)，需要正确处理
            mask_flat = mask.reshape(-1)  # 展平为 (B*T*N,) 或 (B*G*T*N,)
            # 确保 mask_flat 长度与 bands_flat 第一维匹配
            if mask_flat.shape[0] == bands_flat.shape[0]:
                valid = ~mask_flat
                if valid.any():
                    bands_flat = bands_flat[valid]
                else:
                    return torch.tensor(0.0, device=all_bands.device, dtype=all_bands.dtype)

        # 计算频带之间的相关性矩阵
        # 对每个样本计算 K x K 的相关矩阵
        # NOTE: Use a float16-safe eps; PyTorch default eps=1e-12 can underflow to 0 under autocast,
        # producing NaNs when a band feature vector is all-zeros (common for padded/masked tokens).
        bands_norm = F.normalize(bands_flat, dim=1, eps=1e-6)  # L2归一化
        correlation = torch.bmm(bands_norm.transpose(1, 2), bands_norm)  # (B*T*N, K, K)

        # 正交性损失：非对角元素应该接近0
        identity = torch.eye(K, device=correlation.device).unsqueeze(0)
        ortho_loss = ((correlation - identity) ** 2).mean()

        return ortho_loss

    def compute_orthogonality_loss(
        self,
        band_features: Dict,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 综合正交性损失：
        # - filter_loss：仅当滤波器可学习时计算
        # - feature_loss：始终计算（即使 fixed filters 也能鼓励 band 输出差异）
        filters = band_features.get("filters", None)

        if getattr(self, "use_fixed_laplacian", False) or filters is None:
            filter_loss = torch.tensor(0.0, device=self.output_norm.weight.device, dtype=self.output_norm.weight.dtype)
        else:
            filter_loss = self.compute_filter_frequency_orthogonality(filters)

        feature_loss = self.compute_feature_orthogonality(band_features, mask)
        feature_ortho_weight = getattr(self, "feature_ortho_weight", 0.1)
        return filter_loss + feature_ortho_weight * feature_loss

    def compute_energy_balance_loss(
        self,
        band_features: Dict,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        频带能量平衡损失：防止能量集中在少数频带，稳定频带分工。
        """
        all_bands = band_features.get("all_bands", None)
        if all_bands is None:
            return torch.tensor(0.0, device=self.output_norm.weight.device, dtype=self.output_norm.weight.dtype), None

        # all_bands: (B*G, T, N, C, K)
        if all_bands.numel() == 0:
            return torch.tensor(0.0, device=all_bands.device, dtype=all_bands.dtype), None

        bands_flat = all_bands.reshape(-1, all_bands.shape[-2], all_bands.shape[-1])  # (B*G*T*N, C, K)
        if mask is not None:
            mask_flat = mask.reshape(-1)
            if mask_flat.shape[0] == bands_flat.shape[0]:
                valid = ~mask_flat
                if valid.any():
                    bands_flat = bands_flat[valid]
                else:
                    return torch.tensor(0.0, device=all_bands.device, dtype=all_bands.dtype), None

        if bands_flat.numel() == 0:
            return torch.tensor(0.0, device=all_bands.device, dtype=all_bands.dtype), None

        # Energy per band
        energy = bands_flat.pow(2).mean(dim=(0, 1))  # (K,)
        energy = energy / (energy.sum() + 1e-6)
        target = torch.full_like(energy, 1.0 / max(1, energy.numel()))
        loss = ((energy - target) ** 2).mean()
        return loss, energy
    
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_loss: bool = True,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: (B, G, T, N, C) 轨迹特征
            mask: (B, G, T, N) 可选的mask
            return_loss: 是否返回正交性损失
            
        Returns:
            output: 频率增强的特征 (B, G, T, N, C)
            info: 包含频带特征和损失的字典
        """
        B, G, T, N, C = x.shape

        # 边界检查：处理空序列（无目标）
        if N == 0:
            info = {
                'band_features': {'all_bands': torch.zeros(B*G, T, 0, C, self.num_bands, device=x.device, dtype=x.dtype)},
                'gate': torch.zeros(B, G, T, 0, C, device=x.device, dtype=x.dtype),
            }
            if return_loss:
                info['ortho_loss'] = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            return x, info

        # 展平 B, G 维度
        x_flat = x.reshape(B * G, T, N, C)
        
        # 输入归一化
        x_norm = self.input_norm(x_flat)
        
        # 频率分解
        freq_output, band_features = self.freq_filter(x_norm, return_all_bands=True)
        
        # 获取所有频带并融合
        all_bands = band_features['all_bands']  # (B*G, T, N, C, num_bands)
        all_bands_flat = all_bands.reshape(B * G, T, N, -1)  # (B*G, T, N, C*num_bands)
        
        # 频带融合
        fused = self.band_fusion(all_bands_flat)  # (B*G, T, N, C)
        # Use the importance-weighted fused output from the filter to make the adaptive band-importance
        # mechanism actually influence the main path (otherwise `freq_output` becomes dead compute).
        fused = fused + freq_output
        
        # 门控残差连接
        gate_input = torch.cat([x_flat, fused], dim=-1)
        gate = self.output_gate(gate_input)
        
        # 输出
        output = x_flat + gate * fused
        output = self.output_norm(output)
        
        # 恢复形状
        output = output.reshape(B, G, T, N, C)
        
        # 应用mask
        if mask is not None:
            output = output.masked_fill(mask.unsqueeze(-1), 0)
        
        # 准备返回信息
        info = {
            'band_features': band_features,
            'gate': gate.reshape(B, G, T, N, C),
        }
        
        if return_loss:
            info['ortho_loss'] = self.compute_orthogonality_loss(band_features, mask=mask)
            energy_loss, band_energy = self.compute_energy_balance_loss(band_features, mask=mask)
            info['energy_balance_loss'] = energy_loss
            if band_energy is not None:
                info['band_energy'] = band_energy
        
        return output, info


class MultiScaleFrequencyDecomposition(nn.Module):
    """
    多尺度频率分解
    
    在不同的时间尺度上进行频率分解，捕获多粒度的时序模式
    这对于处理不同速度的运动目标特别有效
    """
    
    def __init__(
        self,
        dim: int,
        num_bands: int = 4,
        num_scales: int = 3,
        freq_ortho_metric: str = "dot",
        base_kernel_size: int = 5,
        feature_ortho_weight: float = 0.1,
    ):
        super().__init__()
        self.num_scales = num_scales
        
        # 不同尺度的频率分解
        self.scale_decompositions = nn.ModuleList([
            LearnableFrequencyDecomposition(
                dim=dim,
                num_bands=num_bands,
                kernel_size=base_kernel_size + i * 2,  # 递增的kernel size
                freq_ortho_metric=freq_ortho_metric,
                feature_ortho_weight=feature_ortho_weight,
            )
            for i in range(num_scales)
        ])
        
        # 尺度融合
        self.scale_fusion = nn.Sequential(
            nn.Linear(dim * num_scales, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        
        # 尺度注意力
        self.scale_attention = nn.Sequential(
            nn.Linear(dim, num_scales),
            nn.Softmax(dim=-1)
        )

        # Fuse two fusion paths (attn-weighted vs concat+MLP) with learnable weights.
        # This avoids implicitly doubling the signal by summing two correlated outputs.
        self.fusion_logits = nn.Parameter(torch.zeros(2))

        self.norm = nn.LayerNorm(dim)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: (B, G, T, N, C)
            mask: (B, G, T, N)
        """
        B, G, T, N, C = x.shape

        # 边界检查：空目标保护
        if N == 0:
            info = {
                'scale_outputs': [x for _ in range(self.num_scales)],
                'scale_weights': torch.zeros(B, G, T, 0, self.num_scales, device=x.device, dtype=x.dtype),
                'ortho_loss': torch.tensor(0.0, device=x.device, dtype=x.dtype),
                'all_scale_infos': [],
            }
            return x, info
        
        scale_outputs = []
        all_infos = []
        total_ortho_loss = 0
        
        for i, decomp in enumerate(self.scale_decompositions):
            out, info = decomp(x, mask, return_loss=True)
            scale_outputs.append(out)
            all_infos.append(info)
            total_ortho_loss = total_ortho_loss + info.get('ortho_loss', 0)
        
        # 计算尺度注意力
        x_pool = x.reshape(B * G * T * N, C)
        scale_weights = self.scale_attention(x_pool)  # (B*G*T*N, num_scales)
        # Avoid letting padded tokens contribute gradients to scale_attention.
        # mask: True = padded/invalid
        if mask is not None:
            valid = (~mask).reshape(-1).to(dtype=scale_weights.dtype).unsqueeze(-1)  # (B*G*T*N, 1)
            scale_weights = scale_weights * valid + scale_weights.detach() * (1.0 - valid)
        scale_weights = scale_weights.reshape(B, G, T, N, self.num_scales)
        
        # 加权融合
        stacked = torch.stack(scale_outputs, dim=-1)  # (B, G, T, N, C, num_scales)
        weighted = (stacked * scale_weights.unsqueeze(-2)).sum(dim=-1)
        
        # 也可以用concat+MLP融合
        concat_scales = torch.cat(scale_outputs, dim=-1)  # (B, G, T, N, C*num_scales)
        fused = self.scale_fusion(concat_scales)
        
        # 结合两种融合方式（learnable convex combination）
        fusion_w = torch.softmax(self.fusion_logits, dim=0)  # (2,)
        output = self.norm(fusion_w[0] * weighted + fusion_w[1] * fused)
        
        if mask is not None:
            output = output.masked_fill(mask.unsqueeze(-1), 0)
        
        info = {
            'scale_outputs': scale_outputs,
            'scale_weights': scale_weights,
            'fusion_weights': fusion_w.detach(),
            'ortho_loss': total_ortho_loss / self.num_scales,
            'all_scale_infos': all_infos,
        }
        
        return output, info
