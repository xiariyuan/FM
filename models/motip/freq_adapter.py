# Copyright (c) Ruopeng Gao. All Rights Reserved.
"""
Frequency/Laplacian-style adapter for temporal features.
Acts as a lightweight plug-in before trajectory modeling to inject temporal
high-frequency cues without touching the detection backbone.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class FrequencyAdapter(nn.Module):
    """
    Multi-dilation temporal Laplacian (high-frequency) component with gating.
    Kernels are fixed to [1, -2, 1] with different dilations to better
    approximate a fuller Laplacian response without adding trainable params.
    """

    def __init__(self, dim: int, dilations: Tuple[int, ...] = (1, 2)):
        super().__init__()
        self.convs = nn.ModuleList()
        for d in dilations:
            conv = nn.Conv1d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=3,
                padding=d,
                dilation=d,
                groups=dim,
                bias=False,
            )
            with torch.no_grad():
                k = torch.tensor([1.0, -2.0, 1.0], dtype=torch.float32)
                k = k.view(1, 1, 3).repeat(dim, 1, 1)
                conv.weight.copy_(k)
            conv.weight.requires_grad_(False)  # keep Laplacian fixed
            self.convs.append(conv)
        self.gate = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, G, T, N, C)
            mask: (B, G, T, N) bool, True for padded positions.
        """
        B, G, T, N, C = x.shape
        flat = x.permute(0, 1, 3, 4, 2).reshape(-1, C, T)  # (B*G*N, C, T)
        freq = 0
        for conv in self.convs:
            freq = freq + conv(flat)
        freq = freq.permute(0, 2, 1).reshape(B, G, N, T, C).permute(0, 1, 3, 2, 4).contiguous()
        if mask is not None:
            freq = freq.masked_fill(mask.unsqueeze(-1), 0)
        gate = self.gate(freq)
        fused = x + gate * self.proj(freq)
        return fused
