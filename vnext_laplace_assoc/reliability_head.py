from __future__ import annotations

import torch
from torch import nn


class LaplaceReliabilityHead(nn.Module):
    def __init__(self, pair_dim: int = 6, hidden_dim: int = 16):
        super().__init__()
        hidden_dim = max(int(hidden_dim), 8)
        self.weight_head = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self.reliability_head = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                nn.init.zeros_(module.bias)
        final_weight = self.weight_head[-1]
        final_reliability = self.reliability_head[-1]
        with torch.no_grad():
            final_weight.weight.zero_()
            final_weight.bias.copy_(torch.tensor([1.2, 0.8, 0.4], dtype=final_weight.bias.dtype))
            final_reliability.weight.zero_()
            final_reliability.bias.fill_(0.5)

    def forward(
        self,
        pair_features: torch.Tensor,
        component_scores: torch.Tensor,
    ) -> dict:
        logits = self.weight_head(pair_features)
        weights = torch.softmax(logits, dim=-1)
        reliability = torch.sigmoid(self.reliability_head(pair_features)).squeeze(-1)
        fused = (weights * component_scores).sum(dim=-1)
        motion = component_scores[..., 2]
        final_scores = reliability * fused + (1.0 - reliability) * motion
        return {
            "fused_scores": final_scores.clamp(min=0.0, max=1.0),
            "weights": weights,
            "reliability": reliability,
        }
