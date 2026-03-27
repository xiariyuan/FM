# Copyright (c) 2024. All Rights Reserved.
"""
Frequency-Temporal Transformer (FTT) Module

核心创新点:
1. 分频带时序建模：不同频率使用不同的时序范围
   - 低频：长时序建模（适合长期关联）
   - 高频：短时序建模（适合精确匹配）
2. 跨频带信息交互：让不同频带之间可以交换信息
3. 频率感知的位置编码：根据频带调整位置编码的尺度

理论依据:
- 低频信号变化慢，需要更长的时序窗口来捕获其动态
- 高频信号变化快，过长的时序窗口会引入噪声
- 这种设计类似于小波变换的时频分辨率权衡
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List
from einops import rearrange

# 模块级别检查Mamba可用性
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    Mamba = None
    MAMBA_AVAILABLE = False


class FrequencyAwarePositionalEncoding(nn.Module):
    """
    频率感知的位置编码
    
    不同频带使用不同尺度的位置编码：
    - 低频带：位置编码变化慢（大尺度）
    - 高频带：位置编码变化快（小尺度）
    """
    
    def __init__(
        self,
        dim: int,
        max_len: int = 100,
        num_bands: int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.num_bands = num_bands
        
        # 为每个频带学习一个尺度因子
        self.scale_factors = nn.Parameter(torch.ones(num_bands))
        
        # 基础位置编码
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('base_pe', pe)
        
        # 可学习的频带特定偏移
        self.band_offsets = nn.Parameter(torch.zeros(num_bands, dim))
        
    def forward(self, seq_len: int, band_idx: int) -> torch.Tensor:
        """
        Args:
            seq_len: 序列长度
            band_idx: 频带索引 (0=低频, num_bands-1=高频)
            
        Returns:
            pe: (seq_len, dim) 位置编码
        """
        # NOTE: Do not mutate buffers/state inside forward.
        # During distributed training (DDP) different ranks could see different seq_len and end up with
        # desynchronized buffers if we overwrite `self.base_pe` here. Instead, build a local PE when needed.
        base_pe = self.base_pe
        max_len = int(base_pe.shape[0])
        if seq_len > max_len:
            device = base_pe.device
            dtype = base_pe.dtype
            position = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, self.dim, 2, device=device, dtype=dtype)
                * (-math.log(10000.0) / self.dim)
            )
            pe = torch.zeros(seq_len, self.dim, device=device, dtype=dtype)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            base_pe = pe
            max_len = int(seq_len)

        # 获取尺度因子（低频带用大尺度，高频带用小尺度）
        # 使用sigmoid确保尺度因子为正
        scale = torch.sigmoid(self.scale_factors[band_idx]) * 2 + 0.5  # [0.5, 2.5]
        
        # 调整位置索引
        positions = torch.arange(seq_len, device=base_pe.device, dtype=base_pe.dtype)
        scaled_positions = positions / scale

        # 插值获取位置编码
        # 使用线性插值处理非整数位置
        floor_pos = scaled_positions.floor().long().clamp(0, max_len - 2)
        ceil_pos = (floor_pos + 1).clamp(0, max_len - 1)
        
        floor_weight = (ceil_pos.float() - scaled_positions).unsqueeze(1)
        ceil_weight = (scaled_positions - floor_pos.float()).unsqueeze(1)
        
        pe = floor_weight * base_pe[floor_pos] + ceil_weight * base_pe[ceil_pos]
        
        # 加上频带特定偏移
        pe = pe + self.band_offsets[band_idx]
        
        return pe


class BandSpecificTemporalAttention(nn.Module):
    """
    频带特定的时序注意力
    
    根据频带类型调整注意力的时序范围：
    - 低频带：使用更大的注意力窗口
    - 高频带：使用更小的注意力窗口（局部注意力）
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        band_idx: int = 0,
        num_bands: int = 4,
        max_window_size: int = 30,
        band_window_size: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # 检查维度整除性
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.head_dim = dim // num_heads

        self.band_idx = band_idx
        self.num_bands = num_bands
        self.band_window_size = band_window_size
        
        # 根据频带计算窗口大小
        # 低频带(idx=0)用最大窗口，高频带用最小窗口
        self.window_ratio = 1.0 - (band_idx / max(num_bands - 1, 1)) * 0.7  # [0.3, 1.0]
        self.max_window_size = max_window_size
        
        # 注意力投影
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # 相对位置偏置
        self.relative_position_bias = nn.Parameter(
            torch.zeros(2 * max_window_size - 1, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias, std=0.02)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def _get_attention_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        生成频带特定的注意力mask
        
        低频带：全局注意力（或大窗口）
        高频带：局部注意力（小窗口）
        """
        # 计算当前频带的窗口大小
        if self.band_window_size is not None:
            window_size = int(self.band_window_size)
        else:
            window_size = max(3, int(self.max_window_size * self.window_ratio))
        
        if window_size >= seq_len:
            # 全局注意力，不需要mask
            return None
        
        # 创建局部注意力mask
        mask = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            mask[i, start:end] = False
        
        return mask  # True表示被mask的位置
    
    def _get_relative_position_bias(self, seq_len: int) -> torch.Tensor:
        """获取相对位置偏置"""
        coords = torch.arange(seq_len, device=self.relative_position_bias.device)
        relative_coords = coords[:, None] - coords[None, :]  # (seq_len, seq_len)
        relative_coords = relative_coords + self.max_window_size - 1  # 偏移到正数
        relative_coords = relative_coords.clamp(0, 2 * self.max_window_size - 2)
        
        bias = self.relative_position_bias[relative_coords]  # (seq_len, seq_len, num_heads)
        bias = bias.permute(2, 0, 1)  # (num_heads, seq_len, seq_len)
        
        return bias
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, C) 或 (B*N, T, C)
            mask: (B, T, N) 可选的padding mask
            
        Returns:
            output: 与输入相同形状
        """
        if mask is not None and mask.dtype != torch.bool:
            mask = mask.to(torch.bool)

        padding_mask = None
        if x.dim() == 4:
            B, T, N, C = x.shape
            x = rearrange(x, 'b t n c -> (b n) t c')
            need_reshape = True
            if mask is not None:
                padding_mask = rearrange(mask, 'b t n -> (b n) t')
        else:
            need_reshape = False
            B_N, T, C = x.shape
            if mask is not None:
                padding_mask = mask
        
        # 投影
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 重排为多头形式
        q = rearrange(q, 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(k, 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(v, 'b t (h d) -> b h t d', h=self.num_heads)
        
        # 计算注意力分数
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 添加相对位置偏置
        rel_pos_bias = self._get_relative_position_bias(T)
        attn = attn + rel_pos_bias.unsqueeze(0)
        
        # 应用窗口mask
        window_mask = self._get_attention_mask(T, x.device)
        if window_mask is not None:
            attn = attn.masked_fill(window_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        # 应用padding mask（key侧）- 防止query关注到padding位置
        if padding_mask is not None:
            # key_mask: (B, T) -> (B, 1, 1, T) 用于mask key侧
            attn = attn.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        # Softmax和dropout
        # NOTE:
        # `softmax` over a row of all `-inf` will produce NaNs. This can happen when every key position is masked
        # (e.g., fully padded sequences). Handle those rows explicitly to keep forward/backward well-defined.
        all_masked = torch.isneginf(attn).all(dim=-1, keepdim=True)
        attn = attn.masked_fill(all_masked, 0.0)
        attn = F.softmax(attn, dim=-1)
        # For fully-masked rows, force attention weights to 0 so the output is 0 and gradients don't propagate.
        attn = attn.masked_fill(all_masked, 0.0)
        # Also mask query-side padding so padded queries do not participate (saves useless compute/gradients).
        if padding_mask is not None:
            attn = attn.masked_fill(padding_mask.unsqueeze(1).unsqueeze(-1), 0.0)
        attn = self.dropout(attn)
        
        # 应用注意力
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h t d -> b t (h d)')
        out = self.out_proj(out)
        
        if padding_mask is not None:
            out = out.masked_fill(padding_mask.unsqueeze(-1), 0)
        if need_reshape:
            out = rearrange(out, '(b n) t c -> b t n c', b=B, n=N)
        
        return out


class CrossBandInteraction(nn.Module):
    """
    跨频带信息交互模块
    
    让不同频带之间可以交换信息，实现：
    - 低频指导高频（提供稳定的参考）
    - 高频补充低频（提供细节信息）
    """
    
    def __init__(
        self,
        dim: int,
        num_bands: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_bands = num_bands
        self.num_heads = num_heads
        
        # 跨频带注意力
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # 频带嵌入（用于区分不同频带）
        self.band_embeddings = nn.Parameter(torch.randn(num_bands, dim) * 0.02)
        
        # 交互门控
        self.interaction_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        
        # 输出投影
        self.out_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(
        self,
        band_features: List[torch.Tensor],  # 每个元素: (B, T, N, C)
        band_mask: Optional[torch.Tensor] = None,  # (num_bands,) bool
    ) -> List[torch.Tensor]:
        """
        Args:
            band_features: 各频带特征列表
            
        Returns:
            updated_features: 交互后的各频带特征
        """
        B, T, N, C = band_features[0].shape
        num_bands = len(band_features)
        
        # 将所有频带拼接，加上频带嵌入
        all_bands = []
        for i, feat in enumerate(band_features):
            # 加上频带嵌入
            band_emb = self.band_embeddings[i].unsqueeze(0).unsqueeze(0).unsqueeze(0)
            feat_with_emb = feat + band_emb
            all_bands.append(feat_with_emb)
        
        # 拼接: (B, T, N, num_bands, C) -> (B*T*N, num_bands, C)
        stacked = torch.stack(all_bands, dim=3)
        stacked_flat = rearrange(stacked, 'b t n k c -> (b t n) k c')
        
        # 自注意力实现跨频带交互
        stacked_norm = self.norm1(stacked_flat)
        key_padding_mask = None
        if band_mask is not None:
            # band_mask: True=keep, False=mask
            keep = band_mask.to(device=stacked_norm.device, dtype=torch.bool)
            key_padding_mask = (~keep).unsqueeze(0).expand(stacked_norm.shape[0], -1)
        attended, _ = self.cross_attn(stacked_norm, stacked_norm, stacked_norm, key_padding_mask=key_padding_mask)
        stacked_flat = stacked_flat + attended
        
        # 输出投影
        stacked_flat = stacked_flat + self.out_proj(self.norm2(stacked_flat))
        
        # 分离各频带
        stacked = rearrange(stacked_flat, '(b t n) k c -> b t n k c', b=B, t=T, n=N)
        
        updated_features = []
        for i in range(num_bands):
            updated = stacked[:, :, :, i, :]
            
            # 门控残差
            gate = self.interaction_gate(
                torch.cat([band_features[i], updated], dim=-1)
            )
            final = band_features[i] + gate * (updated - band_features[i])
            updated_features.append(final)
        
        return updated_features


class SpatialFrequencyInteractiveBlock(nn.Module):
    """
    Spatial-Frequency Interactive block (SFI-lite).

    Design goal:
    - Keep the existing frequency-band pipeline intact.
    - Inject a lightweight cross-domain interaction between:
      1) spatial summary (from concatenated multi-band features),
      2) frequency summary (from mean over bands).
    - Use gated residual injection with small initial gain for stability.
    """

    def __init__(
        self,
        dim: int,
        num_bands: int = 4,
        hidden_ratio: float = 2.0,
        alpha_init: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_bands = num_bands
        hidden_dim = max(dim, int(round(dim * float(hidden_ratio))))

        # Build a compact "spatial" representation from all bands.
        self.spatial_fuse = nn.Sequential(
            nn.LayerNorm(dim * num_bands),
            nn.Linear(dim * num_bands, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        # Build a compact "frequency" representation from averaged bands.
        self.freq_fuse = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

        self.s2f_proj = nn.Linear(dim, dim)
        self.f2s_proj = nn.Linear(dim, dim)
        self.s2f_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.f2s_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())

        # Global interaction strength (sigmoid-constrained to (0, 1)).
        alpha_init = float(min(max(alpha_init, 1e-4), 1 - 1e-4))
        alpha_logit_init = math.log(alpha_init / (1.0 - alpha_init))
        self.alpha_logit = nn.Parameter(torch.tensor(alpha_logit_init))

        # Per-band residual gain; start from 0.1 for conservative injection.
        band_gate_init = math.log(0.1 / 0.9)
        self.band_gate_logits = nn.Parameter(torch.full((num_bands,), band_gate_init))
        self.band_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_bands)])

    def forward(self, band_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            band_features: list of length K, each with shape (B, T, N, C)
        Returns:
            updated list with the same shapes.
        """
        if len(band_features) != self.num_bands:
            raise ValueError(
                f"SFI expects {self.num_bands} bands, but got {len(band_features)}."
            )

        # (B, T, N, K, C)
        stacked = torch.stack(band_features, dim=3)
        # Spatial summary: concat all bands on channel.
        spatial_in = torch.cat(band_features, dim=-1)  # (B, T, N, K*C)
        spatial_ctx = self.spatial_fuse(spatial_in)    # (B, T, N, C)
        # Frequency summary: aggregate over bands.
        freq_in = stacked.mean(dim=3)                  # (B, T, N, C)
        freq_ctx = self.freq_fuse(freq_in)             # (B, T, N, C)

        joint = torch.cat([spatial_ctx, freq_ctx], dim=-1)
        s2f = self.s2f_gate(joint) * self.s2f_proj(spatial_ctx)
        f2s = self.f2s_gate(joint) * self.f2s_proj(freq_ctx)

        updated_freq = freq_ctx + s2f
        updated_spatial = spatial_ctx + f2s

        alpha = torch.sigmoid(self.alpha_logit)
        delta = alpha * 0.5 * ((updated_freq - freq_ctx) + (updated_spatial - spatial_ctx))

        out: List[torch.Tensor] = []
        for i in range(self.num_bands):
            band_gain = torch.sigmoid(self.band_gate_logits[i])
            merged = band_features[i] + band_gain * delta
            out.append(self.band_norms[i](merged))
        return out


class FrequencyTemporalTransformer(nn.Module):
    """
    完整的频率-时序Transformer
    
    结合:
    1. 分频带时序注意力
    2. 跨频带信息交互
    3. 频率感知位置编码
    """
    
    def __init__(
        self,
        dim: int,
        num_bands: int = 4,
        num_layers: int = 2,
        num_heads: int = 8,
        max_seq_len: int = 30,
        dropout: float = 0.1,
        use_mamba_for_lowfreq: bool = True,
        band_window_sizes: Optional[List[int]] = None,
        use_spatial_freq_interaction: bool = False,
        sfi_hidden_ratio: float = 2.0,
        sfi_alpha_init: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_bands = num_bands
        self.num_layers = num_layers
        self.use_mamba_for_lowfreq = use_mamba_for_lowfreq
        self.use_spatial_freq_interaction = bool(use_spatial_freq_interaction)
        self.band_window_sizes = band_window_sizes if isinstance(band_window_sizes, list) else None

        if self.band_window_sizes is not None:
            if len(self.band_window_sizes) != num_bands:
                raise ValueError(f"BAND_WINDOW_SIZES length {len(self.band_window_sizes)} != num_bands {num_bands}")
            max_seq_len = max(max_seq_len, max(self.band_window_sizes))
        
        # 频率感知位置编码
        self.pos_encoding = FrequencyAwarePositionalEncoding(
            dim=dim,
            max_len=max_seq_len,
            num_bands=num_bands,
        )
        
        # 各频带的时序注意力层
        self.band_temporal_layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            layer_modules = nn.ModuleList()
            for band_idx in range(num_bands):
                if use_mamba_for_lowfreq and band_idx == 0 and MAMBA_AVAILABLE:
                    # 最低频带使用Mamba（更适合长序列）
                    # Keep normalization consistent with the attention path: apply LayerNorm only in the shared
                    # residual+norm block (self.layer_norms) below. Avoid double LayerNorm on the Mamba path.
                    temporal_module = Mamba(dim)
                else:
                    temporal_module = BandSpecificTemporalAttention(
                        dim=dim,
                        num_heads=num_heads,
                        band_idx=band_idx,
                        num_bands=num_bands,
                        max_window_size=max_seq_len,
                        band_window_size=self.band_window_sizes[band_idx] if self.band_window_sizes is not None else None,
                        dropout=dropout,
                    )
                layer_modules.append(temporal_module)
            self.band_temporal_layers.append(layer_modules)
        
        # 跨频带交互层
        self.cross_band_layers = nn.ModuleList([
            CrossBandInteraction(dim=dim, num_bands=num_bands, num_heads=4, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Optional SFI blocks: inserted after cross-band interaction.
        if self.use_spatial_freq_interaction:
            self.sfi_layers = nn.ModuleList([
                SpatialFrequencyInteractiveBlock(
                    dim=dim,
                    num_bands=num_bands,
                    hidden_ratio=sfi_hidden_ratio,
                    alpha_init=sfi_alpha_init,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ])
        else:
            self.sfi_layers = None
        
        # FFN层
        self.ffn_layers = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim * 4, dim),
                    nn.Dropout(dropout),
                )
                for _ in range(num_bands)
            ])
            for _ in range(num_layers)
        ])
        
        # 归一化层
        self.layer_norms = nn.ModuleList([
            nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_bands)])
            for _ in range(num_layers)
        ])
        
        # 最终融合
        self.final_fusion = nn.Sequential(
            nn.Linear(dim * num_bands, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        
        self.output_norm = nn.LayerNorm(dim)
        # Collecting all intermediate band features is expensive (clones every layer) and is not used by default.
        # Keep it off unless explicitly enabled for debugging/analysis.
        self.collect_intermediate_states = False
        
    def forward(
        self,
        band_features: List[torch.Tensor],  # 各频带特征
        mask: Optional[torch.Tensor] = None,
        band_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            band_features: 各频带特征列表，每个 (B, G, T, N, C)
            mask: (B, G, T, N) padding mask
            
        Returns:
            output: 融合后的特征 (B, G, T, N, C)
            info: 中间信息
        """
        B, G, T, N, C = band_features[0].shape

        # 边界检查：空目标保护
        if N == 0:
            output = band_features[0].reshape(B, G, T, N, C)
            info = {
                'band_features': [f.reshape(B, G, T, N, C) for f in band_features],
                'intermediate_states': [],
            }
            return output, info

        # 展平B, G维度
        band_features = [f.reshape(B * G, T, N, C) for f in band_features]
        if mask is not None:
            mask = mask.reshape(B * G, T, N)
            mask_flat = rearrange(mask, 'bg t n -> (bg n) t')
        else:
            mask_flat = None

        band_active = None
        if band_mask is not None:
            band_active = []
            for i in range(len(band_features)):
                v = band_mask[i]
                if isinstance(v, torch.Tensor):
                    band_active.append(bool(v.item()))
                else:
                    band_active.append(bool(v))

        def _apply_band_mask_inplace(feats: List[torch.Tensor]) -> None:
            if band_active is None:
                return
            for i, active in enumerate(band_active):
                if not active:
                    feats[i] = feats[i] * 0
        
        # 添加位置编码
        for i, feat in enumerate(band_features):
            pe = self.pos_encoding(T, i)  # (T, C)
            band_features[i] = feat + pe.unsqueeze(0).unsqueeze(2)
        _apply_band_mask_inplace(band_features)
        
        intermediate_states: List[List[torch.Tensor]] = []
        
        # 逐层处理
        for layer_idx in range(self.num_layers):
            # 1. 各频带独立的时序建模
            new_band_features = []
            for band_idx, feat in enumerate(band_features):
                if band_active is not None and not band_active[band_idx]:
                    new_band_features.append(feat * 0)
                    continue

                # 时序注意力
                temporal_module = self.band_temporal_layers[layer_idx][band_idx]

                if self.use_mamba_for_lowfreq and band_idx == 0 and MAMBA_AVAILABLE:
                    # Mamba需要(B, T, C)格式
                    # Mamba不支持mask，但强制mask会破坏序列状态
                    # 让Mamba正常处理，依赖后续残差连接的mask过滤
                    feat_flat = rearrange(feat, 'b t n c -> (b n) t c')
                    if mask_flat is not None:
                        feat_flat = feat_flat.masked_fill(mask_flat.unsqueeze(-1), 0)
                    temporal_out = temporal_module(feat_flat)
                    if mask_flat is not None:
                        temporal_out = temporal_out.masked_fill(mask_flat.unsqueeze(-1), 0)
                    temporal_out = rearrange(temporal_out, '(b n) t c -> b t n c', b=B*G, n=N)
                else:
                    temporal_out = temporal_module(feat, mask)
                
                # 残差连接和归一化
                feat = self.layer_norms[layer_idx][band_idx](feat + temporal_out)
                
                # FFN
                feat = feat + self.ffn_layers[layer_idx][band_idx](feat)
                
                new_band_features.append(feat)
            
            band_features = new_band_features
            _apply_band_mask_inplace(band_features)
            
            # 2. 跨频带信息交互
            band_features = self.cross_band_layers[layer_idx](band_features, band_mask=band_mask)
            _apply_band_mask_inplace(band_features)

            # 3. 空间-频率交互（可选）
            if self.sfi_layers is not None:
                band_features = self.sfi_layers[layer_idx](band_features)
                if mask is not None:
                    band_features = [f.masked_fill(mask.unsqueeze(-1), 0) for f in band_features]
                _apply_band_mask_inplace(band_features)

            if self.collect_intermediate_states:
                intermediate_states.append([f.detach().clone() for f in band_features])
        
        # 最终融合
        concat_bands = torch.cat(band_features, dim=-1)  # (B*G, T, N, C*num_bands)
        output = self.final_fusion(concat_bands)  # (B*G, T, N, C)
        output = self.output_norm(output)
        
        # 恢复形状
        output = output.reshape(B, G, T, N, C)
        
        if mask is not None:
            mask = mask.reshape(B, G, T, N)
            output = output.masked_fill(mask.unsqueeze(-1), 0)
        
        # 同样恢复band_features的形状
        band_features = [f.reshape(B, G, T, N, C) for f in band_features]
        
        info = {
            'band_features': band_features,
            'intermediate_states': intermediate_states if self.collect_intermediate_states else [],
        }
        
        return output, info
