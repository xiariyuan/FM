# Copyright (c) Ruopeng Gao. All Rights Reserved.

import torch
import torch.nn as nn

from models.ffn import FFN

try:
    from mamba_ssm import Mamba
except ImportError as _e:  # pragma: no cover
    raise ImportError("Please install mamba-ssm to use the Mamba-based trajectory modeling.") from _e
from models.motip.freq_adapter import FrequencyAdapter


class TrajectoryModeling(nn.Module):
    def __init__(
            self,
            detr_dim: int,
            ffn_dim_ratio: int,
            feature_dim: int,
            use_freq_adapter: bool = False,
    ):
        super().__init__()

        self.detr_dim = detr_dim
        self.ffn_dim_ratio = ffn_dim_ratio
        self.feature_dim = feature_dim
        self.use_freq_adapter = use_freq_adapter
        if detr_dim != feature_dim:
            raise ValueError(f"detr_dim ({detr_dim}) must equal feature_dim ({feature_dim}) for trajectory modeling.")

        self.mamba = Mamba(feature_dim)
        self.mamba_norm = nn.LayerNorm(feature_dim)

        self.freq_adapter = FrequencyAdapter(feature_dim) if self.use_freq_adapter else None
        self.freq_fusion = nn.Sequential(
            nn.LayerNorm(feature_dim * 2),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        ) if self.use_freq_adapter else None

        self.adapter = FFN(
            d_model=detr_dim,
            d_ffn=detr_dim * ffn_dim_ratio,
            activation=nn.GELU(),
        )
        self.norm = nn.LayerNorm(feature_dim)
        self.ffn = FFN(
            d_model=feature_dim,
            d_ffn=feature_dim * ffn_dim_ratio,
            activation=nn.GELU(),
        )
        self.ffn_norm = nn.LayerNorm(feature_dim)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        pass

    def forward(self, seq_info):
        trajectory_features = seq_info["trajectory_features"]  # (B, G, T, N, C)
        trajectory_masks = seq_info.get("trajectory_masks")    # (B, G, T, N)
        _B, _G, _T, _N, _C = trajectory_features.shape

        # 空目标保护：避免Mamba在空batch上报错
        if _N == 0:
            seq_info["trajectory_features"] = trajectory_features
            return seq_info

        # Adapter on DETR features
        appearance_features = trajectory_features + self.adapter(trajectory_features)
        appearance_features = self.norm(appearance_features)

        # Optional frequency/Laplacian enhancement
        if self.freq_adapter is not None:
            freq_enhanced = self.freq_adapter(appearance_features, trajectory_masks)
            freq_delta = freq_enhanced - appearance_features
            fused = self.freq_fusion(torch.cat([appearance_features, freq_delta], dim=-1))
            trajectory_features = appearance_features + fused
        else:
            trajectory_features = appearance_features

        # Temporal modeling with Mamba across T (flatten B*G*N as batch)
        traj_feat_flat = trajectory_features.permute(0, 1, 3, 2, 4).reshape(-1, _T, _C).contiguous()
        if trajectory_masks is not None:
            traj_mask_flat = trajectory_masks.permute(0, 1, 3, 2).reshape(-1, _T).contiguous()
            traj_feat_flat = traj_feat_flat.masked_fill(traj_mask_flat.unsqueeze(-1), 0)
        traj_feat_flat = self.mamba(traj_feat_flat)
        traj_feat_flat = self.mamba_norm(traj_feat_flat)
        if trajectory_masks is not None:
            traj_feat_flat = traj_feat_flat.masked_fill(traj_mask_flat.unsqueeze(-1), 0)
        trajectory_features = traj_feat_flat.view(_B, _G, _N, _T, _C).permute(0, 1, 3, 2, 4).contiguous()

        # FFN refinement
        trajectory_features = trajectory_features + self.ffn(trajectory_features)
        trajectory_features = self.ffn_norm(trajectory_features)
        seq_info["trajectory_features"] = trajectory_features
        return seq_info
