# Copyright (c) 2024. All Rights Reserved.
"""
Frequency-Aware Trajectory Modeling (FA-TM)

这是FA-MOT的核心模块，整合了:
1. Learnable Frequency Decomposition (LFD) - 可学习频率分解
2. Frequency-Temporal Transformer (FTT) - 频率-时序建模
3. Frequency-Guided Association (FGA) - 频率引导关联

整体设计思路:
- 首先将检测特征分解到多个频带
- 然后在每个频带进行时序建模，同时允许跨频带交互
- 最后利用频率信息指导ID关联

创新点:
- 首次在MOT中引入可学习的频率分解
- 设计了频率感知的时序建模（不同频率用不同时序范围）
- 提出遮挡感知的频率权重调整
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List
import math
import warnings

from models.ffn import FFN
from models.misc import label_to_one_hot

# 模块级别检查Mamba可用性（避免每次实例化都检查）
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    Mamba = None
    MAMBA_AVAILABLE = False

# 导入我们设计的三个核心模块
from models.motip.learnable_freq_decomposition import (
    LearnableFrequencyDecomposition,
    MultiScaleFrequencyDecomposition,
)
from models.motip.freq_temporal_transformer import (
    FrequencyTemporalTransformer,
)
from models.motip.freq_guided_association import (
    FrequencyGuidedAssociation,
)

# Optional advanced strategies (occlusion recovery / adaptive bands)
try:
    from models.motip.advanced_strategies import (
        FrequencyGuidedOcclusionRecovery,
        AdaptiveBandSelector,
    )
    _ADV_STRATEGIES_AVAILABLE = True
except Exception as e:
    FrequencyGuidedOcclusionRecovery = None
    AdaptiveBandSelector = None
    _ADV_STRATEGIES_AVAILABLE = False
    warnings.warn(f"[FA-TM] advanced_strategies import failed; related features disabled. Error: {e}")


class FrequencyAwareTrajectoryModeling(nn.Module):
    """
    频率感知的轨迹建模模块
    
    完整的数据流:
    
    输入: trajectory_features (B, G, T, N, C=256)
          ↓
    1. LearnableFrequencyDecomposition
       → 分解为多个频带: [F_0, F_1, ..., F_K-1]
          ↓
    2. FrequencyTemporalTransformer
       → 分频带时序建模 + 跨频带交互
          ↓
    3. 频带融合
       → 融合后的轨迹特征 (B, G, T, N, C)
          ↓
    输出: enhanced_features + freq_info
    
    其中freq_info包含各频带特征，用于后续的FrequencyGuidedAssociation
    """
    
    def __init__(
        self,
        detr_dim: int = 256,
        feature_dim: int = 256,
        ffn_dim_ratio: int = 2,
        # 频率分解参数
        num_bands: int = 4,
        freq_kernel_size: int = 7,
        use_fixed_laplacian: bool = False,
        freq_ortho_metric: str = "dot",
        use_multiscale_freq: bool = False,
        num_freq_scales: int = 3,
        # 时序建模参数
        num_temporal_layers: int = 2,
        temporal_num_heads: int = 8,
        max_seq_len: int = 30,
        use_mamba_for_lowfreq: bool = True,
        use_global_mamba: bool = True,
        band_window_sizes: Optional[List[int]] = None,
        use_spatial_freq_interaction: bool = False,
        sfi_hidden_ratio: float = 2.0,
        sfi_alpha_init: float = 0.1,
        # 其他参数
        dropout: float = 0.1,
        # Advanced strategies
        use_occlusion_recovery: bool = False,
        occlusion_recovery_ratio: float = 0.3,
        use_adaptive_bands: bool = False,
        min_bands: int = 2,
        max_bands: int = 8,
        soft_band_temp: float = 1.0,
        # LFD loss mixing
        lfd_feature_ortho_weight: float = 0.1,
    ):
        super().__init__()
        
        self.detr_dim = detr_dim
        self.feature_dim = feature_dim
        self.num_bands = num_bands
        self.use_multiscale_freq = use_multiscale_freq
        self.use_occlusion_recovery = bool(use_occlusion_recovery) and _ADV_STRATEGIES_AVAILABLE
        self.use_adaptive_bands = bool(use_adaptive_bands) and _ADV_STRATEGIES_AVAILABLE
        self.use_global_mamba = bool(use_global_mamba)
        if detr_dim != feature_dim:
            raise ValueError(f"detr_dim ({detr_dim}) must equal feature_dim ({feature_dim}) for freq-aware modeling.")
        
        # 输入适配器（从DETR特征到我们的特征空间）
        self.input_adapter = FFN(
            d_model=detr_dim,
            d_ffn=detr_dim * ffn_dim_ratio,
            activation=nn.GELU(),
        )
        self.input_norm = nn.LayerNorm(feature_dim)
        
        # 1. 频率分解模块
        if use_multiscale_freq:
            self.freq_decomposition = MultiScaleFrequencyDecomposition(
                dim=feature_dim,
                num_bands=num_bands,
                num_scales=num_freq_scales,
                base_kernel_size=freq_kernel_size,
                freq_ortho_metric=freq_ortho_metric,
                feature_ortho_weight=lfd_feature_ortho_weight,
            )
        else:
            self.freq_decomposition = LearnableFrequencyDecomposition(
                dim=feature_dim,
                num_bands=num_bands,
                kernel_size=freq_kernel_size,
                dropout=dropout,
                freq_ortho_metric=freq_ortho_metric,
                use_fixed_laplacian=use_fixed_laplacian,
                feature_ortho_weight=lfd_feature_ortho_weight,
            )
        
        # 2. 频率-时序Transformer
        self.freq_temporal_transformer = FrequencyTemporalTransformer(
            dim=feature_dim,
            num_bands=num_bands,
            num_layers=num_temporal_layers,
            num_heads=temporal_num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
            use_mamba_for_lowfreq=use_mamba_for_lowfreq,
            band_window_sizes=band_window_sizes,
            use_spatial_freq_interaction=use_spatial_freq_interaction,
            sfi_hidden_ratio=sfi_hidden_ratio,
            sfi_alpha_init=sfi_alpha_init,
        )
        
        # 3. 额外的Mamba时序建模（可选，用于整体时序）
        if MAMBA_AVAILABLE and self.use_global_mamba:
            self.global_mamba = Mamba(feature_dim)
            self.global_mamba_norm = nn.LayerNorm(feature_dim)
            self.has_mamba = True
        else:
            self.has_mamba = False

        # Advanced: occlusion recovery on unknown features
        if self.use_occlusion_recovery:
            self.occlusion_recovery = FrequencyGuidedOcclusionRecovery(
                feature_dim=feature_dim,
                num_bands=num_bands,
                recovery_ratio=occlusion_recovery_ratio,
            )
        else:
            self.occlusion_recovery = None

        # Advanced: adaptive band selector (zero-out high bands)
        if self.use_adaptive_bands:
            max_bands = min(max_bands, num_bands)
            min_bands = min(min_bands, max_bands)
            self.band_selector = AdaptiveBandSelector(
                feature_dim=feature_dim,
                max_bands=max_bands,
                min_bands=min_bands,
                soft_band_temp=soft_band_temp,
            )
        else:
            self.band_selector = None
        
        # 输出FFN
        self.output_ffn = FFN(
            d_model=feature_dim,
            d_ffn=feature_dim * ffn_dim_ratio,
            activation=nn.GELU(),
        )
        self.output_norm = nn.LayerNorm(feature_dim)

        # Forward-time warnings should be attributes (not dynamically created) for DDP/torch.compile friendliness.
        self._warned_unknown_temporal = False
        self._warned_occlusion_recovery_traj = False
        
        # 参数初始化
        self._init_weights()
        
    def _init_weights(self):
        """
        Initialize only lightweight adapter/projection layers owned by this module.

        IMPORTANT:
        Do NOT blindly re-initialize all parameters, otherwise we will destroy:
        - LearnableFrequencyFilter's DoG-like filter initialization
        - Mamba internal parameter initialization
        - Relative position biases / other carefully-initialized submodules
        """
        def _init_linear(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Only init small adapters. Everything else keeps its own init.
        self.input_adapter.apply(_init_linear)
        self.output_ffn.apply(_init_linear)
    
    def forward(
        self,
        seq_info: Dict,
    ) -> Dict:
        """
        Args:
            seq_info: 包含以下键的字典:
                - trajectory_features: (B, G, T, N, C)
                - trajectory_masks: (B, G, T, N), bool, True表示padding
                - unknown_features: (B, G, TU, NU, C) [可选，用于频率分支]
                - unknown_masks: (B, G, TU, NU) [可选]
                - 其他可选键...

        Returns:
            更新后的seq_info，新增:
                - freq_band_features: trajectory各频带特征列表
                - freq_unknown_band_features: unknown各频带特征列表 [如果有unknown]
                - freq_info: 频率相关的中间信息
        """
        trajectory_features = seq_info["trajectory_features"]  # (B, G, T, N, C)
        trajectory_masks = seq_info.get("trajectory_masks")    # (B, G, T, N)
        unknown_features = seq_info.get("unknown_features")    # (B, G, TU, NU, C) or None
        unknown_masks = seq_info.get("unknown_masks")          # (B, G, TU, NU) or None
        TU, NU = 0, 0

        B, G, T, N, C = trajectory_features.shape

        # ============ 处理 unknown（如果存在） ============
        def _coerce_band_shape(tensor: torch.Tensor, expected: Tuple[int, int, int, int, int],
                               fallback: torch.Tensor, name: str) -> torch.Tensor:
            if tensor.shape == expected:
                return tensor
            BxG, T_e, N_e, C_e = expected[0] * expected[1], expected[2], expected[3], expected[4]
            # common case: flattened B*G
            if tensor.dim() == 4 and tensor.shape == (BxG, T_e, N_e, C_e):
                return tensor.reshape(expected)
            # reshape if total elements match
            # NOTE: use math.prod to avoid creating a CPU tensor and to prevent int overflow.
            if tensor.numel() == int(math.prod(expected)):
                return tensor.reshape(expected)
            warnings.warn(f"[FA-TM] {name} shape mismatch {tuple(tensor.shape)} -> {expected}; using fallback.")
            return fallback

        unknown_band_features = None
        if unknown_features is not None:
            _, _, TU, NU, _ = unknown_features.shape

            # 空 unknown 保护：避免 NU==0 时进入频率分解/时序建模
            if TU == 0 or NU == 0:
                unknown_band_features = [unknown_features.clone() for _ in range(self.num_bands)]
            else:
                # 对 unknown 做相同的频率处理
                unk_feat = unknown_features + self.input_adapter(unknown_features)
                unk_feat = self.input_norm(unk_feat)

                # 频率分解（不计算损失，复用trajectory的分解器）
                unk_freq_output, unk_freq_info = self.freq_decomposition(
                    unk_feat,
                    mask=unknown_masks,
                    return_loss=False,  # 只在trajectory上计算正交性损失
                )

                # 提取unknown的频带特征
                if 'all_bands' in unk_freq_info.get('band_features', {}):
                    unk_all_bands = unk_freq_info['band_features']['all_bands']
                    unk_bg, unk_t, unk_n, unk_c, unk_k = unk_all_bands.shape
                    unk_all_bands = unk_all_bands.reshape(B, G, unk_t, unk_n, unk_c, unk_k)
                    unknown_band_features = [
                        unk_all_bands[..., i].contiguous() for i in range(self.num_bands)
                    ]
                else:
                    unknown_band_features = [unk_freq_output.clone() for _ in range(self.num_bands)]

                expected_unk = (B, G, TU, NU, C)
                unknown_band_features = [
                    _coerce_band_shape(f, expected_unk, unknown_features, f"unknown_band_{i}")
                    for i, f in enumerate(unknown_band_features)
                ]

                # Temporal modeling for unknown bands will be applied after adaptive band selection

        # 边界检查：空目标保护（N==0时跳过trajectory频率处理，但仍保留unknown频率分支）
        if N == 0:
            seq_info["freq_band_features"] = [trajectory_features.clone() for _ in range(self.num_bands)]
            if unknown_band_features is not None:
                if TU > 0 and NU > 0:
                    try:
                        _, unk_temporal_info = self.freq_temporal_transformer(
                            unknown_band_features,
                            mask=unknown_masks,
                            band_mask=seq_info.get("band_mask", None),
                        )
                        if 'band_features' in unk_temporal_info:
                            unknown_band_features = unk_temporal_info['band_features']
                    except Exception as e:
                        if not getattr(self, "_warned_unknown_temporal", False):
                            warnings.warn(
                                f"[FA-TM] freq_temporal_transformer failed on unknown bands (N==0 path). "
                                f"Continuing without unknown temporal modeling. Error: {e}"
                            )
                            self._warned_unknown_temporal = True
                seq_info["freq_unknown_band_features"] = unknown_band_features
            seq_info["freq_info"] = {'decomposition_info': {}, 'temporal_info': {}}
            if self.training:
                seq_info['freq_losses'] = {'ortho_loss': trajectory_features.new_tensor(0.0)}
            return seq_info

        # ============ 处理 trajectory ============
        # 1. 输入适配
        features = trajectory_features + self.input_adapter(trajectory_features)
        features = self.input_norm(features)

        # 2. 频率分解
        freq_output, freq_info = self.freq_decomposition(
            features,
            mask=trajectory_masks,
            return_loss=self.training,
        )

        # 提取各频带特征
        if 'all_bands' in freq_info.get('band_features', {}):
            # LFD返回形状: (B*G, T, N, C, num_bands)，需要恢复到(B, G, T, N, C, K)
            all_bands = freq_info['band_features']['all_bands']
            bg, t_len, n_len, c_dim, k_bands = all_bands.shape
            all_bands = all_bands.reshape(B, G, t_len, n_len, c_dim, k_bands)
            band_features = [
                all_bands[..., i].contiguous() for i in range(self.num_bands)
            ]
        else:
            # 如果是MultiScale版本，需要从不同地方获取
            band_features = [freq_output.clone() for _ in range(self.num_bands)]  # 退化情况

        expected_traj = (B, G, T, N, C)
        band_features = [
            _coerce_band_shape(f, expected_traj, trajectory_features, f"traj_band_{i}")
            for i, f in enumerate(band_features)
        ]

        # 3.0 Advanced: adaptive band selection (hard mask bands) BEFORE temporal modeling
        if self.band_selector is not None:
            features_for_complexity = features.reshape(B * G, T, N, C)
            selected_bands, band_info = self.band_selector(
                features=features_for_complexity,
                num_objects=int(N),
                total_bands=self.num_bands,
            )
            band_weights = band_info.get("band_weights", None)
            if band_weights is not None:
                band_weights = band_weights.to(device=features.device, dtype=features.dtype).flatten()
                if band_weights.numel() < self.num_bands:
                    pad = torch.zeros((self.num_bands - band_weights.numel(),), device=features.device, dtype=features.dtype)
                    band_weights = torch.cat([band_weights, pad], dim=0)
                elif band_weights.numel() > self.num_bands:
                    band_weights = band_weights[:self.num_bands]
                for i in range(self.num_bands):
                    band_features[i] = band_features[i] * band_weights[i]
                if unknown_band_features is not None:
                    unknown_band_features = [
                        f * band_weights[idx] for idx, f in enumerate(unknown_band_features)
                    ]

            band_mask = band_info.get("band_mask", None)
            if band_mask is None:
                band_mask = torch.zeros((self.num_bands,), device=features.device, dtype=torch.bool)
                band_mask[:selected_bands] = True
            else:
                band_mask = band_mask.to(device=features.device, dtype=torch.bool).flatten()
                if band_mask.numel() < self.num_bands:
                    pad = torch.zeros((self.num_bands - band_mask.numel(),), device=features.device, dtype=torch.bool)
                    band_mask = torch.cat([band_mask, pad], dim=0)
                elif band_mask.numel() > self.num_bands:
                    band_mask = band_mask[:self.num_bands]

            if int(band_mask.sum().item()) < 1:
                band_mask[0] = True
            for i in range(self.num_bands):
                if not bool(band_mask[i]):
                    band_features[i] = band_features[i] * 0
                    if unknown_band_features is not None:
                        unknown_band_features[i] = unknown_band_features[i] * 0
            seq_info["adaptive_band_info"] = band_info
            seq_info["band_mask"] = band_mask

        # Apply temporal modeling to unknown bands after band selection (if any)
        if unknown_band_features is not None and TU > 0 and NU > 0:
            _, unk_temporal_info = self.freq_temporal_transformer(
                unknown_band_features,
                mask=unknown_masks,
                band_mask=seq_info.get("band_mask", None),
            )
            if 'band_features' in unk_temporal_info:
                expected_unk = (B, G, TU, NU, C)
                unknown_band_features = unk_temporal_info['band_features']
                unknown_band_features = [
                    _coerce_band_shape(f, expected_unk, unknown_features, f"unknown_band_temporal_{i}")
                    for i, f in enumerate(unknown_band_features)
                ]

        # 3. 频率-时序建模
        temporal_output, temporal_info = self.freq_temporal_transformer(
            band_features,
            mask=trajectory_masks,
            band_mask=seq_info.get("band_mask", None),
        )

        # 更新频带特征（时序建模后的）
        if 'band_features' in temporal_info:
            band_features = temporal_info['band_features']
            band_features = [
                _coerce_band_shape(f, expected_traj, trajectory_features, f"traj_band_temporal_{i}")
                for i, f in enumerate(band_features)
            ]
        
        # 4. 可选的全局Mamba时序建模
        if self.has_mamba:
            # 对融合后的特征再做一次全局时序建模
            feat_flat = temporal_output.permute(0, 1, 3, 2, 4).reshape(-1, T, C)
            # Sanity check: expected (B*G*N, T, C)
            if feat_flat.shape != (B * G * N, T, C):
                raise RuntimeError(
                    f"[FA-TM] global_mamba input shape mismatch: got {tuple(feat_flat.shape)}, "
                    f"expected {(B * G * N, T, C)}"
                )
            if trajectory_masks is not None:
                mask_flat = trajectory_masks.permute(0, 1, 3, 2).reshape(-1, T)
                feat_flat = feat_flat.masked_fill(mask_flat.unsqueeze(-1), 0)
            
            global_out = self.global_mamba(feat_flat)
            global_out = self.global_mamba_norm(global_out)
            
            if trajectory_masks is not None:
                global_out = global_out.masked_fill(mask_flat.unsqueeze(-1), 0)
            
            global_out = global_out.view(B, G, N, T, C).permute(0, 1, 3, 2, 4)
            temporal_output = temporal_output + global_out

        # 4.2 Advanced: occlusion recovery for trajectory (optional)
        if self.occlusion_recovery is not None:
            try:
                traj_band_flat = [f.reshape(B * G, T, N, C) for f in band_features]
                recovered_traj, occ_info_traj = self.occlusion_recovery(traj_band_flat)
                recovered_traj = recovered_traj.reshape(B, G, T, N, C)
                # Blend using occlusion scores
                occ_scores = occ_info_traj.get("occlusion_scores", None)
                if occ_scores is not None:
                    occ_scores = occ_scores.reshape(B, G, T, N)
                    temporal_output = temporal_output * (1 - occ_scores.unsqueeze(-1)) + recovered_traj * occ_scores.unsqueeze(-1)
                else:
                    temporal_output = temporal_output + recovered_traj
                seq_info["occlusion_info_traj"] = occ_info_traj
            except Exception as e:
                if not getattr(self, "_warned_occlusion_recovery_traj", False):
                    warnings.warn(
                        f"[FA-TM] occlusion_recovery failed on trajectory bands; continuing without it. Error: {e}"
                    )
                    self._warned_occlusion_recovery_traj = True

        # 5. 输出FFN
        output = temporal_output + self.output_ffn(temporal_output)
        output = self.output_norm(output)
        
        if trajectory_masks is not None:
            output = output.masked_fill(trajectory_masks.unsqueeze(-1), 0)
        
        # 6. Advanced: occlusion recovery for unknown features (optional)
        if self.occlusion_recovery is not None and unknown_band_features is not None and TU > 0 and NU > 0:
            # Flatten B,G for recovery module: (B*G, TU, NU, C)
            unk_recovered, occ_info = self.occlusion_recovery(
                [f.reshape(B * G, f.shape[2], f.shape[3], f.shape[4]) for f in unknown_band_features]
            )
            unk_recovered = unk_recovered.reshape(B, G, TU, NU, C)
            seq_info["unknown_features"] = unk_recovered
            seq_info["occlusion_info"] = occ_info

        # 7. 更新seq_info
        seq_info["trajectory_features"] = output
        seq_info["freq_band_features"] = band_features
        if unknown_band_features is not None:
            seq_info["freq_unknown_band_features"] = unknown_band_features
        seq_info["freq_info"] = {
            'decomposition_info': freq_info,
            'temporal_info': temporal_info,
        }

        # 收集损失项
        if self.training:
            losses = {}
            if 'ortho_loss' in freq_info:
                losses['ortho_loss'] = freq_info['ortho_loss']
            if 'energy_balance_loss' in freq_info:
                losses['energy_balance_loss'] = freq_info['energy_balance_loss']
            seq_info['freq_losses'] = losses

        return seq_info


class FrequencyAwareIDDecoder(nn.Module):
    """
    频率感知的ID解码器
    
    这是对原始IDDecoder的扩展，集成了FrequencyGuidedAssociation
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
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.id_dim = id_dim
        self.num_bands = num_bands
        self.use_freq_guided_association = use_freq_guided_association
        self.num_id_vocabulary = num_id_vocabulary

        # 检查维度整除性
        total_dim = feature_dim + id_dim
        if total_dim % head_dim != 0:
            raise ValueError(f"(feature_dim + id_dim)={total_dim} must be divisible by head_dim={head_dim}")

        # 原始ID词汇表嵌入
        self.word_to_embed = nn.Linear(num_id_vocabulary + 1, id_dim, bias=False)
        self.embed_to_word = nn.Linear(id_dim, num_id_vocabulary + 1, bias=False)
        
        # 频率引导关联模块
        if use_freq_guided_association:
            self.freq_association = FrequencyGuidedAssociation(
                feature_dim=feature_dim,
                id_dim=id_dim,
                num_bands=num_bands,
                num_id_vocabulary=num_id_vocabulary,
                use_occlusion_aware=True,
            )
        
        # Cross-attention for trajectory-to-detection
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim + id_dim,
            num_heads=(feature_dim + id_dim) // head_dim,
            dropout=0.0,
            batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(feature_dim + id_dim)
        
        # FFN
        self.ffn = FFN(
            d_model=feature_dim + id_dim,
            d_ffn=(feature_dim + id_dim) * ffn_dim_ratio,
            activation=nn.GELU(),
        )
        self.ffn_norm = nn.LayerNorm(feature_dim + id_dim)
        
        # 频率特征融合（将多频带信息融入ID解码）
        if use_freq_guided_association:
            self.freq_fusion = nn.Sequential(
                nn.Linear(feature_dim * num_bands, feature_dim * 2),
                nn.GELU(),
                nn.Linear(feature_dim * 2, feature_dim),
            )
            self.freq_gate = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),
                nn.Sigmoid(),
            )
            # Learnable fusion weight between FGA logits and standard decoder logits.
            # Use a logit parameter with sigmoid to keep alpha in (0, 1).
            self.fga_fusion_logit = nn.Parameter(torch.tensor(0.0))
            self._warned_fga_shape_mismatch = False
    
    def forward(
        self,
        seq_info: Dict,
        use_decoder_checkpoint: bool = False,
    ) -> Tuple:
        """
        频率感知的ID解码
        """
        trajectory_features = seq_info["trajectory_features"]
        unknown_features = seq_info["unknown_features"]
        trajectory_id_labels = seq_info["trajectory_id_labels"]
        unknown_id_labels = seq_info.get("unknown_id_labels")
        trajectory_masks = seq_info["trajectory_masks"]
        unknown_masks = seq_info["unknown_masks"]
        
        # 获取频率相关信息
        freq_band_features = seq_info.get("freq_band_features")
        freq_unknown_band_features = seq_info.get("freq_unknown_band_features")
        
        B, G, T, N, C = trajectory_features.shape
        _, _, TU, NU, _ = unknown_features.shape

        # 空unknown保护：直接返回空logits，避免注意力在空序列上报错
        if NU == 0:
            vocab_size = self.num_id_vocabulary + 1
            device = unknown_features.device
            dtype = unknown_features.dtype
            empty_logits = torch.zeros((B, G, TU, 0, vocab_size), device=device, dtype=dtype)
            empty_masks = unknown_masks
            if empty_masks is None:
                empty_masks = torch.zeros((B, G, TU, 0), device=device, dtype=torch.bool)
            labels = unknown_id_labels if self.training else None
            return empty_logits, labels, empty_masks, None
        
        # 如果有频率信息，使用频率引导关联
        id_logits = None
        fga_info = None
        if self.use_freq_guided_association and freq_band_features is not None:
            # Prefer unknown-band features (TU/NU) for association logits if available.
            use_unknown_fga = isinstance(freq_unknown_band_features, (list, tuple)) and len(freq_unknown_band_features) > 0
            if use_unknown_fga:
                band_features_for_fga = freq_unknown_band_features
                band_masks_for_fga = unknown_masks
            else:
                band_features_for_fga = freq_band_features
                band_masks_for_fga = trajectory_masks

            id_logits, fga_info = self.freq_association(
                band_features_for_fga,
                trajectory_masks=band_masks_for_fga,
            )
            
            # 融合频率信息到主特征
            concat_bands = torch.cat(freq_band_features, dim=-1)  # (B, G, T, N, C*num_bands)
            freq_enhanced = self.freq_fusion(concat_bands)  # (B, G, T, N, C)
            
            gate = self.freq_gate(
                torch.cat([trajectory_features, freq_enhanced], dim=-1)
            )
            trajectory_features = trajectory_features + gate * freq_enhanced
        
        # 转换ID标签为嵌入
        trajectory_id_words = label_to_one_hot(
            trajectory_id_labels,
            self.num_id_vocabulary + 1,
            dtype=trajectory_features.dtype
        )
        trajectory_id_embeds = self.word_to_embed(trajectory_id_words)

        # 拼接特征和ID嵌入
        trajectory_embeds = torch.cat([trajectory_features, trajectory_id_embeds], dim=-1)

        # 生成unknown的空ID嵌入
        empty_id_labels = self.num_id_vocabulary * torch.ones(
            unknown_features.shape[:-1],
            dtype=torch.int64,
            device=unknown_features.device
        )
        empty_id_words = label_to_one_hot(
            empty_id_labels,
            self.num_id_vocabulary + 1,
            dtype=unknown_features.dtype
        )
        unknown_id_embeds = self.word_to_embed(empty_id_words)
        unknown_embeds = torch.cat([unknown_features, unknown_id_embeds], dim=-1)
        
        # Cross-attention
        unknown_flat = unknown_embeds.reshape(B * G, -1, unknown_embeds.size(-1))
        trajectory_flat = trajectory_embeds.reshape(B * G, -1, trajectory_embeds.size(-1))
        trajectory_key_padding_mask = (
            trajectory_masks.reshape(B * G, -1) if trajectory_masks is not None else None
        )
        unknown_padding_mask = (
            unknown_masks.reshape(B * G, -1) if unknown_masks is not None else None
        )
        if trajectory_key_padding_mask is not None and trajectory_key_padding_mask.dtype != torch.bool:
            trajectory_key_padding_mask = trajectory_key_padding_mask.to(torch.bool)
        if unknown_padding_mask is not None and unknown_padding_mask.dtype != torch.bool:
            unknown_padding_mask = unknown_padding_mask.to(torch.bool)
        if unknown_padding_mask is not None:
            unknown_flat = unknown_flat.masked_fill(unknown_padding_mask.unsqueeze(-1), 0)
        
        cross_out, _ = self.cross_attn(
            query=unknown_flat,
            key=trajectory_flat,
            value=trajectory_flat,
            key_padding_mask=trajectory_key_padding_mask,
        )
        cross_out = self.cross_attn_norm(unknown_flat + cross_out)
        
        # FFN
        cross_out = self.ffn_norm(cross_out + self.ffn(cross_out))
        if unknown_padding_mask is not None:
            cross_out = cross_out.masked_fill(unknown_padding_mask.unsqueeze(-1), 0)
        
        # 预测ID
        id_embeds = cross_out[..., -self.id_dim:]
        final_logits = self.embed_to_word(id_embeds)
        # 在推理时 unknown 序列长度可能与 trajectory 不同，这里按 unknown 形状还原
        final_logits = final_logits.reshape(B, G, TU, NU, self.num_id_vocabulary + 1)
        final_logits_reshaped = final_logits
        
        # 如果使用了频率引导关联，融合两个预测
        if self.use_freq_guided_association and freq_band_features is not None and id_logits is not None:
            # id_logits来自FGA，final_logits来自standard decoder
            # 可以加权融合；若形状不一致则仅返回标准 decoder 结果
            if id_logits.shape == final_logits_reshaped.shape:
                alpha = torch.sigmoid(self.fga_fusion_logit)  # learnable in (0, 1)
                combined_logits = alpha * id_logits + (1 - alpha) * final_logits_reshaped
                if isinstance(fga_info, dict):
                    fga_info = dict(fga_info)
                    fga_info["fga_fusion_alpha"] = float(alpha.detach().item())
                return combined_logits, unknown_id_labels, unknown_masks, fga_info
            else:
                # 形状不匹配时，退化为标准 decoder 输出，避免 runtime 错误
                if isinstance(fga_info, dict):
                    fga_info = dict(fga_info)
                    if "consistency_loss" in fga_info:
                        fga_info["consistency_loss"] = final_logits_reshaped.new_tensor(0.0)
                    fga_info["fga_skipped_due_to_shape_mismatch"] = True
                if not getattr(self, "_warned_fga_shape_mismatch", False):
                    warnings.warn(
                        "[FrequencyAwareIDDecoder] FGA logits shape mismatch; falling back to standard decoder logits. "
                        "FGA consistency loss is disabled for this forward."
                    )
                    self._warned_fga_shape_mismatch = True
                return final_logits_reshaped, unknown_id_labels, unknown_masks, fga_info

        return final_logits_reshaped, unknown_id_labels, unknown_masks, None
    
    @property
    def dtype(self):
        return self.word_to_embed.weight.dtype


# 便捷函数：创建完整的FA-MOT模块
def build_frequency_aware_modules(config: dict):
    """
    根据配置构建频率感知模块
    
    Args:
        config: 配置字典，应包含:
            - FEATURE_DIM: 特征维度
            - NUM_FREQ_BANDS: 频带数量
            - USE_FREQ_AWARE: 是否使用频率感知
            - ... 其他参数
            
    Returns:
        trajectory_modeling: FrequencyAwareTrajectoryModeling
    """
    use_freq_aware = config.get("USE_FREQ_AWARE", True)
    
    if not use_freq_aware:
        # 返回原始的TrajectoryModeling
        from models.motip.trajectory_modeling import TrajectoryModeling
        return TrajectoryModeling(
            detr_dim=config["DETR_HIDDEN_DIM"],
            ffn_dim_ratio=config["FFN_DIM_RATIO"],
            feature_dim=config["FEATURE_DIM"],
            use_freq_adapter=config.get("USE_FREQ_ADAPTER", False),
        )
    
    return FrequencyAwareTrajectoryModeling(
        detr_dim=config.get("DETR_HIDDEN_DIM", 256),
        feature_dim=config.get("FEATURE_DIM", 256),
        ffn_dim_ratio=config.get("FFN_DIM_RATIO", 2),
        num_bands=config.get("NUM_FREQ_BANDS", config.get("NUM_BANDS", 4)),
        freq_kernel_size=config.get("FREQ_KERNEL_SIZE", 7),
        use_fixed_laplacian=config.get("USE_FIXED_LAPLACIAN", False),
        freq_ortho_metric=config.get("FREQ_ORTHO_METRIC", "dot"),
        use_multiscale_freq=config.get("USE_MULTISCALE_FREQ", False),
        num_freq_scales=config.get("NUM_FREQ_SCALES", 3),
        num_temporal_layers=config.get("NUM_FREQ_TEMPORAL_LAYERS", 2),
        temporal_num_heads=config.get("FREQ_TEMPORAL_HEADS", 8),
        # MAX_SEQ_LEN controls temporal modeling window / positional encoding length.
        # REL_PE_LENGTH is for ID-decoder relative position embedding and should NOT drive FTT length.
        max_seq_len=config.get("MAX_SEQ_LEN", config.get("REL_PE_LENGTH", 30)),
        use_mamba_for_lowfreq=config.get("USE_MAMBA_FOR_LOWFREQ", True),
        use_global_mamba=config.get("USE_GLOBAL_MAMBA", True),
        band_window_sizes=config.get("BAND_WINDOW_SIZES", None),
        use_spatial_freq_interaction=config.get("USE_SPATIAL_FREQ_INTERACTION", False),
        sfi_hidden_ratio=float(config.get("SFI_HIDDEN_RATIO", 2.0)),
        sfi_alpha_init=float(config.get("SFI_ALPHA_INIT", 0.1)),
        dropout=config.get("FREQ_DROPOUT", 0.1),
        lfd_feature_ortho_weight=float(config.get("LFD_FEATURE_ORTHO_WEIGHT", 0.1)),
    )
