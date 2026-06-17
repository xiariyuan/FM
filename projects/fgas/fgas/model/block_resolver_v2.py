from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from projects.fgas.fgas.model.block_resolver import BlockResolverOutput


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weights = mask.unsqueeze(-1).float()
    denom = weights.sum(dim=dim).clamp_min(1.0)
    return (values * weights).sum(dim=dim) / denom


def masked_max(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    expanded_mask = mask.unsqueeze(-1)
    masked = values.masked_fill(~expanded_mask, -1e4)
    pooled = masked.amax(dim=dim)
    valid = mask.any(dim=dim, keepdim=False).unsqueeze(-1)
    return torch.where(valid, pooled, torch.zeros_like(pooled))


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.row_q = nn.LayerNorm(hidden_dim)
        self.col_q = nn.LayerNorm(hidden_dim)
        self.row_to_col = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.col_to_row = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.row_ffn_norm = nn.LayerNorm(hidden_dim)
        self.col_ffn_norm = nn.LayerNorm(hidden_dim)
        self.row_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.col_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(
        self,
        row_tokens: torch.Tensor,
        col_tokens: torch.Tensor,
        row_padding_mask: torch.Tensor,
        col_padding_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        row_attn, _ = self.row_to_col(
            query=self.row_q(row_tokens),
            key=self.col_q(col_tokens),
            value=self.col_q(col_tokens),
            key_padding_mask=col_padding_mask,
            need_weights=False,
        )
        row_tokens = row_tokens + row_attn
        row_tokens = row_tokens + self.row_ffn(self.row_ffn_norm(row_tokens))
        row_tokens = row_tokens.masked_fill(row_padding_mask.unsqueeze(-1), 0.0)

        col_attn, _ = self.col_to_row(
            query=self.col_q(col_tokens),
            key=self.row_q(row_tokens),
            value=self.row_q(row_tokens),
            key_padding_mask=row_padding_mask,
            need_weights=False,
        )
        col_tokens = col_tokens + col_attn
        col_tokens = col_tokens + self.col_ffn(self.col_ffn_norm(col_tokens))
        col_tokens = col_tokens.masked_fill(col_padding_mask.unsqueeze(-1), 0.0)
        return row_tokens, col_tokens


class FGASAssociationResolverV2(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 96,
        stage_embed_dim: int = 16,
        num_stages: int = 4,
        num_heads: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.stage_embedding = nn.Embedding(num_stages, stage_embed_dim)
        edge_in_dim = int(input_dim) + int(stage_embed_dim)
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        node_in_dim = hidden_dim * 2 + 1 + stage_embed_dim
        self.row_init = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.col_init = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList(
            [CrossAttentionBlock(hidden_dim=hidden_dim, num_heads=num_heads) for _ in range(int(num_layers))]
        )
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.row_no_match_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.col_newborn_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        edge_features: torch.Tensor,
        edge_mask: torch.Tensor,
        stage_ids: torch.Tensor,
    ) -> BlockResolverOutput:
        batch_size, row_count, col_count, _ = edge_features.shape
        stage_embed = self.stage_embedding(stage_ids)
        stage_edge = stage_embed.view(batch_size, 1, 1, -1).expand(batch_size, row_count, col_count, -1)
        encoded_edges = self.edge_encoder(torch.cat([edge_features, stage_edge], dim=-1))

        row_present = edge_mask.any(dim=2)
        col_present = edge_mask.any(dim=1)
        row_padding_mask = ~row_present
        col_padding_mask = ~col_present

        row_mean = masked_mean(encoded_edges, edge_mask, dim=2)
        row_max = masked_max(encoded_edges, edge_mask, dim=2)
        row_count_norm = edge_mask.float().sum(dim=2, keepdim=True) / max(col_count, 1)
        row_stage = stage_embed.view(batch_size, 1, -1).expand(batch_size, row_count, -1)
        row_tokens = self.row_init(torch.cat([row_mean, row_max, row_count_norm, row_stage], dim=-1))
        row_tokens = row_tokens.masked_fill(row_padding_mask.unsqueeze(-1), 0.0)

        col_mask = edge_mask.transpose(1, 2)
        col_mean = masked_mean(encoded_edges.transpose(1, 2), col_mask, dim=2)
        col_max = masked_max(encoded_edges.transpose(1, 2), col_mask, dim=2)
        col_count_norm = edge_mask.float().sum(dim=1, keepdim=False).unsqueeze(-1) / max(row_count, 1)
        col_stage = stage_embed.view(batch_size, 1, -1).expand(batch_size, col_count, -1)
        col_tokens = self.col_init(torch.cat([col_mean, col_max, col_count_norm, col_stage], dim=-1))
        col_tokens = col_tokens.masked_fill(col_padding_mask.unsqueeze(-1), 0.0)

        for layer in self.layers:
            row_tokens, col_tokens = layer(
                row_tokens=row_tokens,
                col_tokens=col_tokens,
                row_padding_mask=row_padding_mask,
                col_padding_mask=col_padding_mask,
            )

        row_expand = row_tokens.unsqueeze(2).expand(-1, -1, col_count, -1)
        col_expand = col_tokens.unsqueeze(1).expand(-1, row_count, -1, -1)
        interaction = row_expand * col_expand
        fused = torch.cat([encoded_edges, row_expand, col_expand, interaction], dim=-1)

        edge_logits = self.edge_head(fused).squeeze(-1)
        edge_logits = edge_logits.masked_fill(~edge_mask, -1e4)
        row_no_match_logits = self.row_no_match_head(row_tokens).squeeze(-1).masked_fill(row_padding_mask, -1e4)
        col_newborn_logits = self.col_newborn_head(col_tokens).squeeze(-1).masked_fill(col_padding_mask, -1e4)
        return BlockResolverOutput(
            edge_logits=edge_logits,
            row_no_match_logits=row_no_match_logits,
            col_newborn_logits=col_newborn_logits,
        )
