# Copyright (c) 2024. All Rights Reserved.
"""
Frequency-Aware ID Decoder with Complete Dual-Branch Fusion (FA-IDDecoder v3)

专为ECCV 2026设计的完整双分支融合架构

核心创新点:
1. 双分支独立监督 - 频率分支和标准分支各自有独立的ID损失
2. 注意力融合机制 - 基于置信度和一致性的自适应融合
3. 完整损失设计 - 多目标联合优化
4. 训练-推理一致 - 融合机制在训练和推理时行为一致

架构概览:
                        ┌─────────────────────────────┐
                        │     Input Features          │
                        │  (trajectory + unknown)     │
                        └─────────────┬───────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
                    ▼                                   ▼
    ┌───────────────────────────┐       ┌───────────────────────────┐
    │   Frequency Branch        │       │   Standard Branch         │
    │   ─────────────────       │       │   ───────────────         │
    │   • Multi-band features   │       │   • 6-layer Mamba+Attn    │
    │   • Band-wise ID heads    │       │   • Cross-attention       │
    │   • Band fusion           │       │   • Per-layer outputs     │
    │                           │       │                           │
    │   Output: freq_logits     │       │   Output: std_logits      │
    │   Loss: L_freq            │       │   Loss: L_std (aux)       │
    └─────────────┬─────────────┘       └─────────────┬─────────────┘
                  │                                   │
                  └─────────────┬─────────────────────┘
                                │
                                ▼
                ┌───────────────────────────────┐
                │   Attention-based Fusion      │
                │   ─────────────────────────   │
                │   • Confidence estimation     │
                │   • Consistency gating        │
                │   • Adaptive weighting        │
                │                               │
                │   Output: fused_logits        │
                │   Loss: L_fusion              │
                └───────────────────────────────┘

总损失: L = L_std + λ1*L_freq + λ2*L_fusion + λ3*L_consist + λ4*L_ortho
"""

import torch
import einops
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
from typing import Tuple, Dict, Optional, List
from torch.utils.checkpoint import checkpoint

from models.misc import _get_clones, label_to_one_hot


def _meshgrid_ij(x, y):
    """torch.meshgrid with backward-compatible indexing arg."""
    try:
        return torch.meshgrid(x, y, indexing='ij')
    except TypeError:
        return torch.meshgrid(x, y)


def _safe_l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """
    L2-normalize with an eps that is safe under mixed precision.

    PyTorch's default eps=1e-12 can underflow to 0 in float16, causing 0/0 -> NaN on padded/all-zero vectors.
    """
    return F.normalize(x, p=2.0, dim=dim, eps=float(eps))
from models.ffn import FFN

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    Mamba = None
    MAMBA_AVAILABLE = False

# Optional confidence calibration
try:
    from models.motip.advanced_strategies import FrequencyAwareConfidenceCalibration
    _CALIB_AVAILABLE = True
except Exception as e:
    FrequencyAwareConfidenceCalibration = None
    _CALIB_AVAILABLE = False
    warnings.warn(f"[FA-IDDecoderV2] advanced_strategies import failed; confidence calibration disabled. Error: {e}")


class MambaBlock(nn.Module):
    """Mamba block for sequence modeling."""
    def __init__(self, d_model: int):
        super().__init__()
        if not MAMBA_AVAILABLE:
            raise ImportError("Please install mamba-ssm")
        self.mamba = Mamba(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0)
        out = self.mamba(x)
        out = self.norm(out)
        if padding_mask is not None:
            out = out.masked_fill(padding_mask.unsqueeze(-1), 0)
        return out


class FrequencyBranch(nn.Module):
    """
    频率分支（带上下文）：使用 trajectory 频带特征 + 轨迹标签进行匹配，
    再把相似度散射到 vocab，从而在随机标签设定下可学习。
    """

    def __init__(
        self,
        feature_dim: int,
        num_bands: int,
        num_id_vocabulary: int,
        temperature_init: float = 0.07,
        neg_large: float = -1e4,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_bands = num_bands
        self.num_id_vocabulary = num_id_vocabulary
        self.neg_large = float(neg_large)

        self.band_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.GELU(),
                nn.LayerNorm(feature_dim),
            )
            for _ in range(num_bands)
        ])

        self.band_attention = nn.Linear(feature_dim, 1)

        init = math.log(1.0 / max(1e-6, float(temperature_init)))
        self.logit_scale = nn.Parameter(torch.ones(num_bands) * init)

        self.newborn_bias = nn.Parameter(torch.tensor(0.0))
        self.newborn_scale = nn.Parameter(torch.tensor(1.0))

        self.global_confidence = nn.Linear(feature_dim, 1)
        self.final_confidence = nn.Sequential(
            nn.Linear(num_bands + 2, max(1, feature_dim // 4)),
            nn.GELU(),
            nn.Linear(max(1, feature_dim // 4), 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        unknown_band_features: List[torch.Tensor],  # (B,G,TU,NU,C)
        trajectory_band_features: List[torch.Tensor],  # (B,G,T,N,C)
        trajectory_id_labels: torch.Tensor,  # (B,G,T,N)
        trajectory_masks: torch.Tensor,  # (B,G,T,N)
        trajectory_times: torch.Tensor,  # (B,G,T,N)
        unknown_times: torch.Tensor,  # (B,G,TU,NU)
        band_mask: Optional[torch.Tensor] = None,  # (num_bands,) bool
    ) -> Tuple[torch.Tensor, Dict]:
        if len(unknown_band_features) == 0:
            raise ValueError("unknown_band_features cannot be empty")
        if len(trajectory_band_features) == 0:
            raise ValueError("trajectory_band_features cannot be empty")

        B, G, TU, NU, C = unknown_band_features[0].shape
        vocab_size = self.num_id_vocabulary + 1
        device = unknown_band_features[0].device
        dtype = unknown_band_features[0].dtype

        # 空目标保护：NU==0时返回空tensor
        if NU == 0:
            empty_logits = torch.zeros((B, G, TU, 0, vocab_size), device=device, dtype=dtype)
            empty_conf = torch.zeros((B, G, TU, 0), device=device, dtype=dtype)
            info = {
                'band_logits': [empty_logits for _ in range(self.num_bands)],
                'band_confidence': torch.zeros((B, G, TU, 0, self.num_bands), device=device, dtype=dtype),
                'freq_logits': empty_logits,
                'freq_confidence': empty_conf,
                'band_weights': torch.zeros((B, G, TU, 0, self.num_bands), device=device, dtype=dtype),
                'features': torch.zeros((B, G, TU, 0, C), device=device, dtype=dtype),
            }
            return empty_logits, info

        if trajectory_masks is not None and trajectory_masks.dtype != torch.bool:
            trajectory_masks = trajectory_masks.to(torch.bool)

        BG = B * G
        T = trajectory_band_features[0].shape[2]
        N = trajectory_band_features[0].shape[3]

        # Track labels are constant per track, but tracks may start later (early frames padded).
        # Use the first valid timestep per track to avoid accidentally taking a masked (-1) label at t=0.
        if trajectory_masks is not None:
            valid_track = (~trajectory_masks).any(dim=2)  # (B, G, N)
            first_valid_t = (~trajectory_masks).float().argmax(dim=2)  # (B, G, N)
            gather_idx = first_valid_t.unsqueeze(2)  # (B, G, 1, N)
            track_labels = torch.gather(trajectory_id_labels, dim=2, index=gather_idx).squeeze(2)  # (B, G, N)
            track_labels = track_labels.masked_fill(~valid_track, -1)
        else:
            valid_track = torch.ones((B, G, N), dtype=torch.bool, device=device)
            track_labels = trajectory_id_labels[:, :, 0, :].clone()
        track_labels = track_labels.clamp(min=0, max=vocab_size - 1)

        traj_masks_flat = trajectory_masks.reshape(BG, T * N) if trajectory_masks is not None else None
        traj_times_flat = trajectory_times.reshape(BG, T * N)
        unk_times_flat = unknown_times.reshape(BG, TU * NU)

        band_logits_list = []
        band_conf_list = []
        band_feat_list = []

        neg_large = self.neg_large

        if band_mask is None:
            band_mask = torch.ones((self.num_bands,), dtype=torch.bool, device=device)
        else:
            band_mask = band_mask.to(device=device, dtype=torch.bool)

        for k in range(self.num_bands):
            if not bool(band_mask[k]):
                # Skip masked bands: fill with neg_large logits and zero confidence/features
                empty_logits = torch.full((B, G, TU, NU, vocab_size), neg_large, device=device, dtype=dtype)
                band_logits_list.append(empty_logits)
                band_conf_list.append(torch.zeros((B, G, TU, NU), device=device, dtype=dtype))
                band_feat_list.append(torch.zeros((B, G, TU, NU, C), device=device, dtype=dtype))
                continue
            traj = trajectory_band_features[k].reshape(BG, T * N, C)
            unk = unknown_band_features[k].reshape(BG, TU * NU, C)

            if traj_masks_flat is not None:
                traj = traj.masked_fill(traj_masks_flat.unsqueeze(-1), 0.0)
            traj = _safe_l2_normalize(traj, dim=-1, eps=1e-6)
            unk = _safe_l2_normalize(unk, dim=-1, eps=1e-6)

            scale = self.logit_scale[k].exp().clamp(max=100.0)
            sim = torch.bmm(unk, traj.transpose(1, 2)) * scale  # (BG, TU*NU, T*N)

            if traj_masks_flat is not None:
                sim = sim.masked_fill(traj_masks_flat.unsqueeze(1), neg_large)
            # Causal masking:
            # Disallow attending to the same frame or future frames to avoid train/infer leakage.
            # This matches the standard branch (cross-attn) causal mask semantics.
            causal = traj_times_flat.unsqueeze(1) >= unk_times_flat.unsqueeze(2)
            sim = sim.masked_fill(causal, neg_large)

            sim_tn = sim.view(BG, TU * NU, T, N)
            track_sim = sim_tn.max(dim=2).values  # (BG, TU*NU, N)

            logits = torch.full((BG, TU * NU, vocab_size), neg_large, device=device, dtype=dtype)

            idx = track_labels.reshape(BG, N)
            idx = idx.clamp(min=0, max=vocab_size - 1)
            idx = idx.unsqueeze(1).expand(-1, TU * NU, -1)

            src = track_sim
            if src.dtype != logits.dtype:
                src = src.to(dtype=logits.dtype)
            valid_mask = valid_track.reshape(BG, N)
            src = src.masked_fill(~valid_mask.unsqueeze(1), neg_large)

            if hasattr(logits, "scatter_reduce_"):
                logits.scatter_reduce_(dim=-1, index=idx, src=src, reduce="amax", include_self=True)
            else:
                # Fallback: manual max-reduce for older PyTorch
                for n in range(N):
                    idx_n = idx[:, :, n]
                    src_n = src[:, :, n]
                    gathered = logits.gather(dim=-1, index=idx_n.unsqueeze(-1)).squeeze(-1)
                    updated = torch.maximum(gathered, src_n)
                    logits.scatter_(dim=-1, index=idx_n.unsqueeze(-1), src=updated.unsqueeze(-1))

            max_sim = track_sim.max(dim=-1).values
            newborn_scale = F.softplus(self.newborn_scale) + 1e-6
            newborn_logit = self.newborn_bias - newborn_scale * max_sim
            if newborn_logit.dtype != logits.dtype:
                newborn_logit = newborn_logit.to(dtype=logits.dtype)
            # Avoid in-place modification after scatter_reduce to keep autograd stable
            logits = torch.cat([logits[:, :, :-1], newborn_logit.unsqueeze(-1)], dim=-1)

            logits = logits.view(B, G, TU, NU, vocab_size)
            band_logits_list.append(logits)

            probs = F.softmax(logits, dim=-1)
            conf = probs.max(dim=-1).values  # (B,G,TU,NU)
            band_conf_list.append(conf)

            band_feat_list.append(self.band_projs[k](unknown_band_features[k]))

        band_attention = torch.cat([self.band_attention(f) for f in band_feat_list], dim=-1)
        if band_mask is not None:
            mask_logits = (~band_mask).view(1, 1, 1, 1, -1).to(band_attention.device)
            band_attention = band_attention.masked_fill(mask_logits, float("-inf"))
        band_weights = F.softmax(band_attention, dim=-1)

        freq_logits = None
        for k in range(self.num_bands):
            w = band_weights[..., k:k+1]
            if freq_logits is None:
                freq_logits = band_logits_list[k] * w
            else:
                freq_logits = freq_logits + band_logits_list[k] * w

        fused_features = None
        for k in range(self.num_bands):
            w = band_weights[..., k:k+1]
            if fused_features is None:
                fused_features = band_feat_list[k] * w
            else:
                fused_features = fused_features + band_feat_list[k] * w

        global_conf = torch.sigmoid(self.global_confidence(fused_features)).squeeze(-1)
        all_band_conf = torch.stack(band_conf_list, dim=-1)
        band_conf_max = all_band_conf.max(dim=-1).values
        conf_in = torch.cat([
            global_conf.unsqueeze(-1),
            band_conf_max.unsqueeze(-1),
            all_band_conf,
        ], dim=-1)
        final_conf = self.final_confidence(conf_in).squeeze(-1)

        info = {
            'band_logits': band_logits_list,
            'band_confidence': all_band_conf,
            'band_weights': band_weights,
            'freq_logits': freq_logits,
            'freq_confidence': final_conf,
            'features': fused_features,
        }

        return freq_logits, info


class AttentionFusion(nn.Module):
    """
    基于注意力的双分支融合模块
    
    创新点:
    1. 置信度感知融合 - 根据各分支的置信度动态调整权重
    2. 一致性门控 - 两分支预测一致时增加融合权重
    3. 残差融合 - 保持梯度流动的稳定性
    """
    
    def __init__(
        self,
        feature_dim: int,
        vocab_size: int,
        num_bands: int = 4,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.vocab_size = vocab_size
        
        # 融合权重预测网络
        # 输入：两个分支的logits + 特征 + 置信度
        fusion_input_dim = vocab_size * 2 + feature_dim + num_bands + 2  # +2 for two confidence scores
        
        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_input_dim, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, 2),  # 输出两个分支的权重
        )
        
        # 一致性评估网络
        self.consistency_net = nn.Sequential(
            nn.Linear(vocab_size * 2, vocab_size),
            nn.GELU(),
            nn.Linear(vocab_size, 1),
            nn.Sigmoid(),
        )
        
        # 可学习的先验权重
        self.prior_weights = nn.Parameter(torch.tensor([0.5, 0.5]))
        
        # 残差门控
        self.residual_gate = nn.Sequential(
            nn.Linear(vocab_size, vocab_size // 2),
            nn.GELU(),
            nn.Linear(vocab_size // 2, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        freq_logits: torch.Tensor,      # (B, G, TU, NU, vocab)
        std_logits: torch.Tensor,        # (B, G, TU, NU, vocab)
        features: torch.Tensor,          # (B, G, TU, NU, C)
        freq_confidence: torch.Tensor,   # (B, G, TU, NU)
        band_confidence: torch.Tensor,   # (B, G, TU, NU, num_bands)
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Returns:
            fused_logits: (B, G, TU, NU, vocab)
            info: 融合信息字典
        """
        B, G, TU, NU, vocab = freq_logits.shape
        
        # 计算标准分支的置信度（基于预测熵）
        std_probs = F.softmax(std_logits, dim=-1)
        std_probs = torch.clamp(std_probs, min=1e-6, max=1.0)  # 防止数值不稳定
        std_entropy = -(std_probs * std_probs.log()).sum(dim=-1)  # (B, G, TU, NU)
        std_confidence = 1.0 / (1.0 + std_entropy)  # 熵越低，置信度越高
        
        # 评估两分支的一致性
        # Use normalized log-probabilities as fusion inputs for numerical stability.
        # Raw logits can be scaled up (e.g. logit_scale.exp()), making MLP inputs unstable.
        freq_logp = F.log_softmax(freq_logits, dim=-1)
        std_logp = F.log_softmax(std_logits, dim=-1)
        consistency_input = torch.cat([freq_logp, std_logp], dim=-1)
        consistency = self.consistency_net(consistency_input).squeeze(-1)  # (B, G, TU, NU)
        
        # 准备融合网络的输入
        fusion_input = torch.cat([
            freq_logp,
            std_logp,
            features,
            band_confidence,
            freq_confidence.unsqueeze(-1),
            std_confidence.unsqueeze(-1),
        ], dim=-1)
        
        # 预测融合权重
        raw_weights = self.fusion_net(fusion_input)  # (B, G, TU, NU, 2)
        
        # 结合先验权重
        prior = F.softmax(self.prior_weights, dim=0)
        fusion_weights = F.softmax(raw_weights, dim=-1) * 0.7 + prior * 0.3
        
        # 应用融合权重
        fused_logits = (
            fusion_weights[..., 0:1] * freq_logits + 
            fusion_weights[..., 1:2] * std_logits
        )
        
        # 残差连接：基于一致性的门控
        residual_gate = self.residual_gate(fused_logits).squeeze(-1)  # (B, G, TU, NU)
        # When two branches disagree (low consistency), we should rely more on the learned fusion
        # to resolve the conflict. When they agree, a simple average is sufficient and more stable.
        gate = (1.0 - consistency) * residual_gate
        
        # 最终输出：融合结果 + 残差
        # 一致性高 -> 更偏向平均；一致性低 -> 更偏向学习到的融合
        avg_logits = 0.5 * (freq_logits + std_logits)
        final_logits = gate.unsqueeze(-1) * fused_logits + (1 - gate.unsqueeze(-1)) * avg_logits
        
        info = {
            'fusion_weights': fusion_weights,
            'consistency': consistency,
            'freq_confidence': freq_confidence,
            'std_confidence': std_confidence,
            'residual_gate': residual_gate,
            'gate': gate,
        }
        
        return final_logits, info


class FrequencyAwareIDDecoderV3(nn.Module):
    """
    完整的频率感知双分支ID解码器 (V3 - ECCV版本)
    
    特点:
    1. 双分支独立监督 - 频率分支和标准分支各自有ID损失
    2. 注意力融合 - 基于置信度和一致性的自适应融合
    3. 完整损失设计 - 支持多目标联合优化
    4. 训练-推理一致 - 融合机制始终生效
    """
    
    def __init__(
        self,
        feature_dim: int,
        id_dim: int,
        ffn_dim_ratio: int,
        num_layers: int,
        head_dim: int,
        num_id_vocabulary: int,
        rel_pe_length: int,
        use_aux_loss: bool,
        use_shared_aux_head: bool,
        # 频率相关参数
        num_bands: int = 4,
        use_freq_guided_association: bool = True,
        use_learnable_fusion: bool = True,
        # Standard-branch self modeling
        use_mamba_self_attn: bool = True,
        # 损失权重（可通过config覆盖）
        freq_loss_weight: float = 1.0,
        fusion_loss_weight: float = 1.0,
        # Training options
        label_smoothing: float = 0.0,
        use_confidence_calibration: bool = False,
        calibration_strength: float = 0.5,
        min_confidence: float = 0.1,
        use_newborn_head: bool = False,
        newborn_head_dim: int = 128,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.id_dim = id_dim
        self.ffn_dim_ratio = ffn_dim_ratio
        self.num_layers = num_layers
        self.head_dim = head_dim

        # 检查维度整除性
        total_dim = feature_dim + id_dim
        if total_dim % head_dim != 0:
            raise ValueError(f"(feature_dim + id_dim)={total_dim} must be divisible by head_dim={head_dim}")
        self.n_heads = total_dim // head_dim

        self.num_id_vocabulary = num_id_vocabulary
        self.rel_pe_length = rel_pe_length
        self.num_bands = num_bands
        self.use_freq_guided_association = use_freq_guided_association
        self.use_learnable_fusion = use_learnable_fusion
        self.use_mamba_self_attn = bool(use_mamba_self_attn)
        
        self.use_aux_loss = use_aux_loss
        self.use_shared_aux_head = use_shared_aux_head
        
        self.freq_loss_weight = freq_loss_weight
        self.fusion_loss_weight = fusion_loss_weight
        self.label_smoothing = float(label_smoothing)
        self.use_confidence_calibration = bool(use_confidence_calibration) and _CALIB_AVAILABLE
        self.calibration_strength = float(calibration_strength)
        self.use_newborn_head = bool(use_newborn_head)
        self.newborn_head_dim = int(newborn_head_dim)
        # Optional inference-time calibration head. Keep the attribute defined for
        # all code paths so older checkpoints / partial feature toggles do not hit
        # an AttributeError during forward().
        self.confidence_calibrator = None
        
        vocab_size = num_id_vocabulary + 1
        
        # ============ 标准分支 (Standard Branch) ============
        # ID词汇表嵌入
        self.word_to_embed = nn.Linear(vocab_size, id_dim, bias=False)
        embed_to_word = nn.Linear(id_dim, vocab_size, bias=False)
        
        if use_aux_loss and not use_shared_aux_head:
            self.embed_to_word_layers = _get_clones(embed_to_word, num_layers)
        else:
            self.embed_to_word_layers = nn.ModuleList([embed_to_word for _ in range(num_layers)])
        
        # 相对位置编码
        self.rel_pos_embeds = nn.Parameter(
            torch.zeros((num_layers, rel_pe_length * 2 - 1, self.n_heads), dtype=torch.float32)
        )
        t_idxs = torch.arange(rel_pe_length, dtype=torch.int64)
        curr_t_idxs, traj_t_idxs = _meshgrid_ij(t_idxs, t_idxs)
        # Map relative positions to non-negative indices [0, 2*L-2].
        self.register_buffer('rel_pos_map', curr_t_idxs - traj_t_idxs + (rel_pe_length - 1))
        
        # Self-attention (configurable): Mamba (default) or Transformer self-attention.
        if self.use_mamba_self_attn:
            self.self_mamba_layers = _get_clones(
                MambaBlock(feature_dim + id_dim), num_layers - 1
            )
            self.self_attn_layers = None
        else:
            self.self_mamba_layers = None
            self_attn = nn.MultiheadAttention(
                embed_dim=feature_dim + id_dim,
                num_heads=self.n_heads,
                dropout=0.0,
                batch_first=True,
                add_zero_attn=False,
            )
            self.self_attn_layers = _get_clones(self_attn, num_layers - 1)
        self.self_attn_norm_layers = _get_clones(
            nn.LayerNorm(feature_dim + id_dim), num_layers - 1
        )
        
        # Cross-attention
        cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim + id_dim,
            num_heads=self.n_heads,
            dropout=0.0,
            batch_first=True,
            add_zero_attn=True,
        )
        cross_attn_norm = nn.LayerNorm(feature_dim + id_dim)
        self.cross_attn_layers = _get_clones(cross_attn, num_layers)
        self.cross_attn_norm_layers = _get_clones(cross_attn_norm, num_layers)
        
        # FFN
        ffn = FFN(
            d_model=feature_dim + id_dim,
            d_ffn=(feature_dim + id_dim) * ffn_dim_ratio,
            activation=nn.GELU(),
        )
        ffn_norm = nn.LayerNorm(feature_dim + id_dim)
        self.ffn_layers = _get_clones(ffn, num_layers)
        self.ffn_norm_layers = _get_clones(ffn_norm, num_layers)
        
        # ============ 频率分支 (Frequency Branch) ============
        if use_freq_guided_association:
            # 特征增强
            self.freq_feature_enhance = nn.Sequential(
                nn.Linear(feature_dim * num_bands, feature_dim * 2),
                nn.GELU(),
                nn.Linear(feature_dim * 2, feature_dim),
            )
            self.freq_gate = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),
                nn.Sigmoid(),
            )
            
            # 频率分支
            neg_large = -1e4 if self.label_smoothing <= 0.0 else -20.0
            self.frequency_branch = FrequencyBranch(
                feature_dim=feature_dim,
                num_bands=num_bands,
                num_id_vocabulary=num_id_vocabulary,
                neg_large=neg_large,
            )
            
            # 注意力融合
            self.attention_fusion = AttentionFusion(
                feature_dim=feature_dim,
                vocab_size=vocab_size,
                num_bands=num_bands,
            )

            # Optional confidence calibrator (trained end-to-end)
            if self.use_confidence_calibration:
                self.confidence_calibrator = FrequencyAwareConfidenceCalibration(
                    num_bands=num_bands,
                    calibration_strength=calibration_strength,
                    min_confidence=min_confidence,
                )
            else:
                self.confidence_calibrator = None

        # Newborn head (optional): separate head for newborn logit
        if self.use_newborn_head:
            self.newborn_head = nn.Sequential(
                nn.Linear(feature_dim, max(1, self.newborn_head_dim)),
                nn.GELU(),
                nn.Linear(max(1, self.newborn_head_dim), 1),
            )
        else:
            self.newborn_head = None

        # Forward-time warnings should be attributes (not dynamically created) for DDP/torch.compile friendliness.
        self._warned_triplet_newborn_filter = False
        
        # 参数初始化
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize only a small set of linear projection layers.

        IMPORTANT:
        Do NOT blanket re-initialize all parameters, otherwise we will destroy:
        - Mamba internal initialization (critical for stability)
        - learned scalars/prior logits (fusion/logit scales)
        - any submodule-specific initializations
        """
        # Keep rel_pos_embeds initialized to zeros (neutral bias).

        # word_to_embed / embed_to_word benefit from stable Xavier init.
        nn.init.xavier_uniform_(self.word_to_embed.weight)
        # embed_to_word_layers may share the same module; init once is enough.
        if isinstance(self.embed_to_word_layers, nn.ModuleList) and len(self.embed_to_word_layers) > 0:
            emb2word = self.embed_to_word_layers[0]
            if isinstance(emb2word, nn.Linear):
                nn.init.xavier_uniform_(emb2word.weight)
    
    def forward(
        self,
        seq_info: Dict,
        use_decoder_checkpoint: bool = False,
    ) -> Tuple:
        """
        完整的双分支前向传播
        
        Returns:
            outputs: 根据训练/推理模式返回不同内容
            - 训练时: (all_logits, all_labels, all_masks, extra_info)
              * all_logits 包含标准分支各层 + 频率分支 + 融合结果
            - 推理时: (fused_logits, labels, masks, extra_info)
        """
        trajectory_features = seq_info["trajectory_features"]
        unknown_features = seq_info["unknown_features"]
        trajectory_id_labels = seq_info["trajectory_id_labels"]
        unknown_id_labels = seq_info.get("unknown_id_labels")
        trajectory_times = seq_info["trajectory_times"]
        unknown_times = seq_info["unknown_times"]
        trajectory_masks = seq_info["trajectory_masks"]
        unknown_masks = seq_info["unknown_masks"]
        
        freq_band_features = seq_info.get("freq_band_features")
        freq_unknown_band_features = seq_info.get("freq_unknown_band_features")
        band_mask = seq_info.get("band_mask", None)

        B, G, T, N, C = trajectory_features.shape
        _, _, TU, NU, _ = unknown_features.shape

        # 空unknown保护：直接返回空logits，避免后续注意力/Mamba报错
        if NU == 0:
            vocab_size = self.num_id_vocabulary + 1
            device = unknown_features.device
            dtype = unknown_features.dtype
            empty_logits = torch.zeros((B, G, TU, 0, vocab_size), device=device, dtype=dtype)
            empty_masks = unknown_masks
            if empty_masks is None:
                empty_masks = torch.zeros((B, G, TU, 0), device=device, dtype=torch.bool)
            extra_info = {"empty_unknown": True}

            if self.training:
                std_layers = self.num_layers if self.use_aux_loss else 1
                all_logits_list = [empty_logits for _ in range(std_layers)]
                all_labels_list = [unknown_id_labels] * std_layers
                all_masks_list = [empty_masks] * std_layers
                all_weights_list = [1.0] * std_layers

                has_freq = self.use_freq_guided_association and freq_unknown_band_features is not None
                if has_freq:
                    # 频率分支不做 CE 监督，仅保留融合分支（若有）
                    all_logits_list.append(empty_logits)
                    all_labels_list.append(unknown_id_labels)
                    all_masks_list.append(empty_masks)
                    all_weights_list.append(self.fusion_loss_weight)

                all_logits = torch.cat(all_logits_list, dim=0)
                all_masks = torch.cat(all_masks_list, dim=0)
                all_labels = (
                    torch.cat(all_labels_list, dim=0) if unknown_id_labels is not None else None
                )

                extra_info["loss_weights"] = all_weights_list
                extra_info["num_std_layers"] = std_layers
                extra_info["has_freq_branch"] = has_freq
                if has_freq:
                    extra_info["consistency_loss"] = torch.tensor(0.0, device=device, dtype=dtype)

                return all_logits, all_labels, all_masks, extra_info

            return empty_logits, unknown_id_labels, empty_masks, extra_info

        extra_info = {}
        freq_logits = None
        freq_branch_info = None

        # ============ 频率分支处理 ============
        if self.use_freq_guided_association and freq_unknown_band_features is not None:
            # 1. 特征增强（使用trajectory的频带特征增强trajectory）
            concat_bands = torch.cat(freq_band_features, dim=-1)
            freq_enhanced = self.freq_feature_enhance(concat_bands)
            gate = self.freq_gate(torch.cat([trajectory_features, freq_enhanced], dim=-1))
            trajectory_features_enhanced = trajectory_features + gate * freq_enhanced

            # 2. 频率分支预测（使用unknown的频带特征）
            freq_logits, freq_branch_info = self.frequency_branch(
                unknown_band_features=freq_unknown_band_features,
                trajectory_band_features=freq_band_features,
                trajectory_id_labels=trajectory_id_labels,
                trajectory_masks=trajectory_masks,
                trajectory_times=trajectory_times,
                unknown_times=unknown_times,
                band_mask=band_mask,
            )
            extra_info['freq_branch_info'] = freq_branch_info

            # 使用增强后的特征
            trajectory_features = trajectory_features_enhanced
        
        # ============ 标准分支处理 ============
        # 转换ID标签为嵌入
        trajectory_id_embeds = self.id_label_to_embed(trajectory_id_labels)
        unknown_id_embeds = self.generate_empty_id_embed(unknown_features)
        
        trajectory_embeds = torch.cat([trajectory_features, trajectory_id_embeds], dim=-1)
        unknown_embeds = torch.cat([unknown_features, unknown_id_embeds], dim=-1)
        
        # 准备attention masks
        self_attn_key_padding_mask = einops.rearrange(unknown_masks, "b g t n -> (b g t) n").contiguous()
        cross_attn_key_padding_mask = einops.rearrange(trajectory_masks, "b g t n -> (b g) (t n)").contiguous()
        
        _trajectory_times_flatten = einops.rearrange(trajectory_times, "b g t n -> (b g) (t n)")
        _unknown_times_flatten = einops.rearrange(unknown_times, "b g t n -> (b g) (t n)")
        # Causal constraint: unknown tokens at time t should not attend to trajectory tokens at time >= t.
        cross_attn_mask_bool = _trajectory_times_flatten[:, None, :] >= _unknown_times_flatten[:, :, None]  # (bg, Lq, Lk)
        cross_attn_mask_bool = einops.repeat(
            cross_attn_mask_bool,
            "bg lq lk -> (bg n_heads) lq lk",
            n_heads=self.n_heads,
        ).contiguous()
        
        # 相对位置编码
        rel_pe_idx_pairs = torch.stack([
            torch.stack(
                _meshgrid_ij(_unknown_times_flatten[_], _trajectory_times_flatten[_]), dim=-1
            )
            for _ in range(len(_trajectory_times_flatten))
        ], dim=0)
        rel_pe_idx_pairs = rel_pe_idx_pairs.to(trajectory_features.device)
        rel_pe_idx_pairs = rel_pe_idx_pairs.clamp(0, self.rel_pe_length - 1)
        rel_pos_map = self.rel_pos_map.to(trajectory_features.device)
        rel_pe_idxs = rel_pos_map[rel_pe_idx_pairs[..., 0], rel_pe_idx_pairs[..., 1]]
        
        # Keep self-attn key padding mask as bool for Mamba
        if self_attn_key_padding_mask is not None and self_attn_key_padding_mask.dtype != torch.bool:
            self_attn_key_padding_mask = self_attn_key_padding_mask.to(torch.bool)
        if cross_attn_key_padding_mask is not None and cross_attn_key_padding_mask.dtype != torch.bool:
            cross_attn_key_padding_mask = cross_attn_key_padding_mask.to(torch.bool)
        # Build a single additive attention mask (float) that includes both:
        # - causal mask (time >= t)
        # - key padding mask (padded trajectory tokens)
        #
        # Then we can pass `key_padding_mask=None` to MultiheadAttention to avoid dtype-mismatch warnings and
        # keep masking semantics explicit and stable.
        cross_attn_mask = torch.zeros_like(cross_attn_mask_bool, dtype=self.dtype)
        cross_attn_mask = cross_attn_mask.masked_fill(cross_attn_mask_bool, float("-inf"))

        if cross_attn_key_padding_mask is not None:
            bg = int(_trajectory_times_flatten.shape[0])
            Lq = int(_unknown_times_flatten.shape[1])
            Lk = int(_trajectory_times_flatten.shape[1])
            if cross_attn_key_padding_mask.shape != (bg, Lk):
                raise ValueError(
                    f"cross_attn_key_padding_mask shape mismatch: got {tuple(cross_attn_key_padding_mask.shape)}, "
                    f"expected {(bg, Lk)}"
                )

            key_pad = cross_attn_key_padding_mask.unsqueeze(1).expand(bg, Lq, Lk)  # (bg, Lq, Lk)
            key_pad = einops.repeat(
                key_pad,
                "bg lq lk -> (bg n_heads) lq lk",
                n_heads=self.n_heads,
            ).contiguous()
            key_pad_mask = torch.zeros_like(cross_attn_mask, dtype=self.dtype).masked_fill(key_pad, float("-inf"))
            cross_attn_mask = cross_attn_mask + key_pad_mask
            cross_attn_key_padding_mask = None
        
        # 逐层解码
        std_layer_logits = []
        
        for layer in range(self.num_layers):
            if use_decoder_checkpoint:
                unknown_embeds = checkpoint(
                    self._forward_a_layer,
                    layer,
                    unknown_embeds, trajectory_embeds,
                    self_attn_key_padding_mask, cross_attn_key_padding_mask,
                    cross_attn_mask, rel_pe_idxs,
                    use_reentrant=False,
                )
            else:
                unknown_embeds = self._forward_a_layer(
                    layer=layer,
                    unknown_embeds=unknown_embeds,
                    trajectory_embeds=trajectory_embeds,
                    self_attn_key_padding_mask=self_attn_key_padding_mask,
                    cross_attn_key_padding_mask=cross_attn_key_padding_mask,
                    cross_attn_mask=cross_attn_mask,
                    rel_pe_idx=rel_pe_idxs,
                )
            
            layer_logits = self.embed_to_word_layers[layer](unknown_embeds[..., -self.id_dim:])
            std_layer_logits.append(layer_logits)
        
        # Optional newborn head (override last logit)
        newborn_logit = None
        if self.use_newborn_head and self.newborn_head is not None:
            newborn_logit = self.newborn_head(unknown_features).squeeze(-1)  # (B, G, TU, NU)
            if unknown_masks is not None:
                newborn_logit = newborn_logit.masked_fill(unknown_masks, 0.0)
            std_layer_logits = [
                torch.cat([logits[..., :-1], newborn_logit.unsqueeze(-1)], dim=-1)
                for logits in std_layer_logits
            ]
            if freq_logits is not None:
                freq_logits = torch.cat([freq_logits[..., :-1], newborn_logit.unsqueeze(-1)], dim=-1)
            extra_info["newborn_logit"] = newborn_logit

        # 标准分支最终输出
        std_logits = std_layer_logits[-1]  # (B, G, TU, NU, vocab)

        # 若未启用辅助损失，仅保留最后一层输出用于监督，减少显存与噪声
        if not self.use_aux_loss:
            std_layer_logits = [std_logits]
        
        # ============ 双分支融合 ============
        if freq_logits is not None and freq_logits.shape == std_logits.shape:
            # 获取融合所需的信息
            freq_confidence = freq_branch_info['freq_confidence']
            band_confidence = freq_branch_info['band_confidence'][:, :, -1:, :NU, :] if TU == 1 else freq_branch_info['band_confidence'][:, :, :TU, :NU, :]
            
            if self.use_learnable_fusion:
                # 注意力融合（可学习）
                fused_logits, fusion_info = self.attention_fusion(
                    freq_logits=freq_logits,
                    std_logits=std_logits,
                    features=unknown_features,
                    freq_confidence=freq_confidence,
                    band_confidence=band_confidence,
                )
            else:
                # 固定权重融合（便于消融）
                fused_logits = 0.5 * freq_logits + 0.5 * std_logits
                fusion_info = {
                    'fusion_weights': torch.full(
                        (B, G, TU, NU, 2), 0.5, device=std_logits.device, dtype=std_logits.dtype
                    ),
                    'consistency': None,
                    'freq_confidence': freq_confidence,
                    'std_confidence': None,
                    'residual_gate': None,
                }
            extra_info['fusion_info'] = fusion_info
        else:
            # 无法融合时，使用标准分支
            fused_logits = std_logits
            extra_info['fusion_skipped'] = True

        # ============ 置信度校准（可选） ============
        # Calibration is an inference-time post-processing step. Applying it during training changes the
        # logits scale/nonlinearly and can destabilize optimization. Keep training on raw logits.
        confidence_calibrator = getattr(self, "confidence_calibrator", None)
        if (not self.training) and self.use_confidence_calibration and confidence_calibrator is not None:
            band_logits = None
            if isinstance(freq_branch_info, dict):
                band_logits = freq_branch_info.get("band_logits", None)
            if isinstance(band_logits, list) and len(band_logits) > 0:
                calibration_factor = confidence_calibrator.compute_calibration_factor(
                    band_logits=band_logits,
                    fused_logits=fused_logits,
                )
                # Apply calibration to fused logits (inference only)
                probs = F.softmax(fused_logits, dim=-1)
                uniform = torch.full_like(probs, 1.0 / probs.shape[-1])
                probs = calibration_factor.unsqueeze(-1) * probs + (1.0 - calibration_factor.unsqueeze(-1)) * uniform
                fused_logits = probs.clamp(min=1e-6).log()

                extra_info["calibration_factor"] = calibration_factor
                extra_info["calibration_applied"] = True
            else:
                extra_info["calibration_applied"] = False
        else:
            extra_info["calibration_applied"] = False
        
        # ============ 构建返回值 ============
        if self.training:
            # 训练时：返回所有需要计算损失的logits
            # 结构: [std_layer_0, std_layer_1, ..., std_layer_n-1, freq_logits, fused_logits]
            
            all_logits_list = list(std_layer_logits)  # 拷贝，避免后续 append 污染原列表
            all_labels_list = [unknown_id_labels] * len(std_layer_logits)
            all_masks_list = [unknown_masks] * len(std_layer_logits)
            all_weights_list = [1.0] * len(std_layer_logits)  # 标准分支权重
            
            if freq_logits is not None:
                # 可选：对频率分支本身做监督
                if self.freq_loss_weight is not None and float(self.freq_loss_weight) > 0:
                    all_logits_list.append(freq_logits)
                    all_labels_list.append(unknown_id_labels)
                    all_masks_list.append(unknown_masks)
                    all_weights_list.append(self.freq_loss_weight)
                # 融合结果监督
                all_logits_list.append(fused_logits)
                all_labels_list.append(unknown_id_labels)
                all_masks_list.append(unknown_masks)
                all_weights_list.append(self.fusion_loss_weight)
            
            # 拼接所有logits
            all_logits = torch.cat(all_logits_list, dim=0)
            all_labels = torch.cat(all_labels_list, dim=0) if unknown_id_labels is not None else None
            all_masks = torch.cat(all_masks_list, dim=0)
            
            extra_info['loss_weights'] = all_weights_list
            extra_info['num_std_layers'] = len(std_layer_logits)
            extra_info['has_freq_branch'] = freq_logits is not None
            extra_info['triplet_embeddings'] = unknown_embeds[..., -self.id_dim:]
            # IMPORTANT: filter newborn samples from triplet supervision.
            # In this codebase, newborn label is `num_id_vocabulary` (the last class, vocab_size = num_id_vocabulary + 1).
            # Treat them as invalid (-1) so TripletLoss won't pull different newborn detections together.
            triplet_labels = unknown_id_labels
            if triplet_labels is not None:
                newborn_idx = int(self.num_id_vocabulary)
                try:
                    triplet_labels = triplet_labels.clone()
                    triplet_labels = triplet_labels.masked_fill(triplet_labels == newborn_idx, -1)
                except Exception as e:
                    if not getattr(self, "_warned_triplet_newborn_filter", False):
                        warnings.warn(
                            f"[FrequencyAwareIDDecoderV2] Failed to filter newborn labels for triplet supervision; "
                            f"falling back to raw labels. Error: {e}"
                        )
                        self._warned_triplet_newborn_filter = True
                    # fallback: keep original labels if anything goes wrong
                    triplet_labels = unknown_id_labels
            extra_info['triplet_labels'] = triplet_labels
            extra_info['triplet_masks'] = unknown_masks
            
            # 计算一致性损失
            if freq_logits is not None:
                consistency_loss = 0.5 * (
                    self._compute_consistency_loss(freq_logits, std_logits.detach(), masks=unknown_masks) +
                    self._compute_consistency_loss(freq_logits.detach(), std_logits, masks=unknown_masks)
                )
                extra_info['consistency_loss'] = consistency_loss
            
            return all_logits, all_labels, all_masks, extra_info
        else:
            # 推理时：返回融合后的结果
            return fused_logits, unknown_id_labels, unknown_masks, extra_info
    
    def _forward_a_layer(
        self,
        layer: int,
        unknown_embeds: torch.Tensor,
        trajectory_embeds: torch.Tensor,
        self_attn_key_padding_mask: torch.Tensor,
        cross_attn_key_padding_mask: torch.Tensor,
        cross_attn_mask: torch.Tensor,
        rel_pe_idx: torch.Tensor,
    ):
        B, G, T, N, _ = trajectory_embeds.shape
        _, _, TU, NU, _ = unknown_embeds.shape
        
        # Self-attention (Mamba or Transformer)
        if layer > 0:
            self_unknown_embeds = einops.rearrange(unknown_embeds, "b g t n c -> (b g t) n c").contiguous()
            if self_attn_key_padding_mask is not None:
                self_unknown_embeds = self_unknown_embeds.masked_fill(
                    self_attn_key_padding_mask.unsqueeze(-1), 0
                )

            if self.use_mamba_self_attn:
                if self.self_mamba_layers is None:
                    raise RuntimeError("[FrequencyAwareIDDecoderV2] self_mamba_layers is None but use_mamba_self_attn=True")
                self_out = self.self_mamba_layers[layer - 1](
                    self_unknown_embeds, padding_mask=self_attn_key_padding_mask
                )
            else:
                if self.self_attn_layers is None:
                    raise RuntimeError("[FrequencyAwareIDDecoderV2] self_attn_layers is None but use_mamba_self_attn=False")
                self_out, _ = self.self_attn_layers[layer - 1](
                    query=self_unknown_embeds,
                    key=self_unknown_embeds,
                    value=self_unknown_embeds,
                    need_weights=False,
                    key_padding_mask=self_attn_key_padding_mask,
                )
                if self_attn_key_padding_mask is not None:
                    self_out = self_out.masked_fill(self_attn_key_padding_mask.unsqueeze(-1), 0)
            self_out = self_unknown_embeds + self_out
            self_out = self.self_attn_norm_layers[layer - 1](self_out)
            if self_attn_key_padding_mask is not None:
                self_out = self_out.masked_fill(self_attn_key_padding_mask.unsqueeze(-1), 0)
            unknown_embeds = einops.rearrange(self_out, "(b g t) n c -> b g t n c", b=B, g=G, t=TU)
        
        # Cross-attention
        cross_unknown_embeds = einops.rearrange(unknown_embeds, "b g t n c -> (b g) (t n) c").contiguous()
        cross_trajectory_embeds = einops.rearrange(trajectory_embeds, "b g t n c -> (b g) (t n) c").contiguous()
        
        # 相对位置编码
        rel_pe_mask = self.rel_pos_embeds[layer][rel_pe_idx]
        rel_pe_bias = einops.rearrange(rel_pe_mask, "bg l1 l2 n -> (bg n) l1 l2").to(cross_attn_mask.dtype)
        # Handle add_zero_attn=True by padding a zero-bias column for the extra key token.
        if rel_pe_bias.shape[-1] != cross_attn_mask.shape[-1]:
            pad = int(cross_attn_mask.shape[-1] - rel_pe_bias.shape[-1])
            if pad > 0:
                rel_pe_bias = F.pad(rel_pe_bias, (0, pad), value=0.0)
            else:
                rel_pe_bias = rel_pe_bias[..., :cross_attn_mask.shape[-1]]
        cross_attn_mask_with_rel_pe = cross_attn_mask + rel_pe_bias
        
        cross_out, _ = self.cross_attn_layers[layer](
            query=cross_unknown_embeds, 
            key=cross_trajectory_embeds, 
            value=cross_trajectory_embeds,
            need_weights=False,
            key_padding_mask=cross_attn_key_padding_mask,
            attn_mask=cross_attn_mask_with_rel_pe,
        )
        cross_out = cross_unknown_embeds + cross_out
        cross_out = self.cross_attn_norm_layers[layer](cross_out)
        
        # FFN
        cross_out = cross_out + self.ffn_layers[layer](cross_out)
        cross_out = self.ffn_norm_layers[layer](cross_out)
        
        unknown_embeds = einops.rearrange(cross_out, "(b g) (t n) c -> b g t n c", b=B, g=G, t=TU)
        
        return unknown_embeds
    
    def _compute_consistency_loss(
        self,
        freq_logits: torch.Tensor,
        std_logits: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        计算双分支一致性损失 (JS Divergence)，对无效mask做过滤，量级稳定。
        """
        eps = 1e-8

        freq_logp = F.log_softmax(freq_logits / temperature, dim=-1)
        std_logp = F.log_softmax(std_logits / temperature, dim=-1)

        freq_p = freq_logp.exp()
        std_p = std_logp.exp()

        avg_p = (0.5 * (freq_p + std_p)).clamp(min=eps)
        avg_logp = avg_p.log()

        kl_freq = (freq_p * (freq_logp - avg_logp)).sum(dim=-1)
        kl_std = (std_p * (std_logp - avg_logp)).sum(dim=-1)
        js = 0.5 * (kl_freq + kl_std)  # (B, G, T, N)

        if masks is not None:
            valid = (~masks).float()
            js = (js * valid).sum() / (valid.sum() + eps)
        else:
            js = js.mean()

        return js
    
    def id_label_to_embed(self, id_labels):
        id_words = label_to_one_hot(id_labels, self.num_id_vocabulary + 1, dtype=self.dtype)
        id_embeds = self.word_to_embed(id_words)
        return id_embeds
    
    def generate_empty_id_embed(self, unknown_features):
        _shape = unknown_features.shape[:-1]
        empty_id_labels = self.num_id_vocabulary * torch.ones(_shape, dtype=torch.int64, device=unknown_features.device)
        empty_id_embeds = self.id_label_to_embed(id_labels=empty_id_labels)
        return empty_id_embeds
    
    @property
    def dtype(self):
        return self.word_to_embed.weight.dtype


# Backward-compatible alias for configs/imports expecting V2.
class FrequencyAwareIDDecoderV2(FrequencyAwareIDDecoderV3):
    """Alias of FrequencyAwareIDDecoderV3 for compatibility."""
    pass
