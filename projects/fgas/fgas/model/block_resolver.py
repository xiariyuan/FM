from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn


@dataclass
class BlockResolverOutput:
    edge_logits: torch.Tensor
    row_no_match_logits: torch.Tensor
    col_newborn_logits: torch.Tensor | None = None


class FGASBlockResolver(nn.Module):
    """
    Minimal block-level resolver.

    Input:
    - edge_features: [B, R, C, F]
    - edge_mask: [B, R, C]

    Output:
    - edge_logits: [B, R, C]
    - row_no_match_logits: [B, R]

    This model is intentionally small enough for a first clean run while still
    being stronger than the old pair-wise linear scorer.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, stage_embed_dim: int = 8, num_stages: int = 4) -> None:
        super().__init__()
        self.stage_embedding = nn.Embedding(num_stages, stage_embed_dim)
        fused_dim = int(input_dim) + int(stage_embed_dim)
        self.edge_encoder = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.edge_head = nn.Linear(hidden_dim * 3, 1)
        self.row_no_match_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        edge_features: torch.Tensor,
        edge_mask: torch.Tensor,
        stage_ids: torch.Tensor,
    ) -> BlockResolverOutput:
        batch_size, row_count, col_count, _ = edge_features.shape
        stage_embed = self.stage_embedding(stage_ids).view(batch_size, 1, 1, -1).expand(batch_size, row_count, col_count, -1)
        encoded = self.edge_encoder(torch.cat([edge_features, stage_embed], dim=-1))

        mask = edge_mask.unsqueeze(-1).float()
        row_den = mask.sum(dim=2).clamp_min(1.0)
        col_den = mask.sum(dim=1).clamp_min(1.0)
        row_ctx = (encoded * mask).sum(dim=2, keepdim=True) / row_den.unsqueeze(2)
        col_ctx = (encoded * mask).sum(dim=1, keepdim=True) / col_den.unsqueeze(1)
        row_ctx = row_ctx.expand(-1, -1, col_count, -1)
        col_ctx = col_ctx.expand(-1, row_count, -1, -1)
        fused = torch.cat([encoded, row_ctx, col_ctx], dim=-1)

        edge_logits = self.edge_head(fused).squeeze(-1)
        edge_logits = edge_logits.masked_fill(~edge_mask, -1e4)

        row_summary = (encoded * mask).sum(dim=2) / row_den
        row_no_match_logits = self.row_no_match_head(row_summary).squeeze(-1)
        return BlockResolverOutput(
            edge_logits=edge_logits,
            row_no_match_logits=row_no_match_logits,
            col_newborn_logits=None,
        )


STAGE_NAME_TO_ID: Dict[str, int] = {
    "primary": 0,
    "recovery": 1,
    "unconfirmed": 2,
    "lifecycle": 3,
}
