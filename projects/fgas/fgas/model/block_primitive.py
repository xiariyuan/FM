from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn

from projects.fgas.fgas.model.block_resolver import STAGE_NAME_TO_ID
from projects.fgas.fgas.model.block_resolver_v3_trackquery import TrackDetCrossLayer, masked_max, masked_mean


MODEL_FAMILY = "fgas_block_primitive"


@dataclass
class BlockPrimitiveOutput:
    edge_logits: torch.Tensor
    row_no_match_logits: torch.Tensor
    col_newborn_logits: torch.Tensor
    block_confidence_logits: torch.Tensor


class FGASAmbiguityBlockPrimitive(nn.Module):
    """
    Joint ambiguity-block association primitive.

    This model treats a local ambiguity block as the learning unit and predicts:
    - edge match logits
    - row no-match logits
    - col newborn logits
    - block confidence logit for local takeover vs fallback
    """

    def __init__(
        self,
        input_dim: int,
        row_context_dim: int,
        col_context_dim: int,
        hidden_dim: int = 128,
        stage_embed_dim: int = 16,
        num_stages: int = 4,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.stage_embedding = nn.Embedding(int(num_stages), int(stage_embed_dim))
        self.edge_encoder = nn.Sequential(
            nn.Linear(int(input_dim) + int(stage_embed_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
        )
        self.row_context_encoder = nn.Sequential(
            nn.Linear(int(row_context_dim) + int(stage_embed_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )
        self.col_context_encoder = nn.Sequential(
            nn.Linear(int(col_context_dim) + int(stage_embed_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )
        self.row_edge_proj = nn.Sequential(
            nn.Linear(int(hidden_dim) * 2, int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )
        self.col_edge_proj = nn.Sequential(
            nn.Linear(int(hidden_dim) * 2, int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )
        self.layers = nn.ModuleList(
            [
                TrackDetCrossLayer(
                    hidden_dim=int(hidden_dim),
                    num_heads=int(num_heads),
                    dropout=float(dropout),
                )
                for _ in range(int(num_layers))
            ]
        )
        self.edge_head = nn.Sequential(
            nn.Linear(int(hidden_dim) * 4, int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), 1),
        )
        self.row_no_match_head = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), 1),
        )
        self.col_newborn_head = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), 1),
        )
        self.block_confidence_head = nn.Sequential(
            nn.Linear(int(hidden_dim) * 6, int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(
        self,
        edge_features: torch.Tensor,
        edge_mask: torch.Tensor,
        stage_ids: torch.Tensor,
        row_context: torch.Tensor,
        col_context: torch.Tensor,
    ) -> BlockPrimitiveOutput:
        batch_size, row_count, col_count, _ = edge_features.shape
        stage_embed = self.stage_embedding(stage_ids)
        stage_edge = stage_embed.view(batch_size, 1, 1, -1).expand(batch_size, row_count, col_count, -1)
        stage_row = stage_embed.view(batch_size, 1, -1).expand(batch_size, row_count, -1)
        stage_col = stage_embed.view(batch_size, 1, -1).expand(batch_size, col_count, -1)

        encoded_edges = self.edge_encoder(torch.cat([edge_features, stage_edge], dim=-1))
        row_present = edge_mask.any(dim=2)
        col_present = edge_mask.any(dim=1)
        row_padding_mask = ~row_present
        col_padding_mask = ~col_present

        row_mean = masked_mean(encoded_edges, edge_mask, dim=2)
        row_max = masked_max(encoded_edges, edge_mask, dim=2)
        col_mask = edge_mask.transpose(1, 2)
        col_mean = masked_mean(encoded_edges.transpose(1, 2), col_mask, dim=2)
        col_max = masked_max(encoded_edges.transpose(1, 2), col_mask, dim=2)

        row_tokens = self.row_context_encoder(torch.cat([row_context, stage_row], dim=-1)) + self.row_edge_proj(
            torch.cat([row_mean, row_max], dim=-1)
        )
        col_tokens = self.col_context_encoder(torch.cat([col_context, stage_col], dim=-1)) + self.col_edge_proj(
            torch.cat([col_mean, col_max], dim=-1)
        )
        row_tokens = row_tokens.masked_fill(row_padding_mask.unsqueeze(-1), 0.0)
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
        fused = torch.cat([encoded_edges, row_expand, col_expand, row_expand * col_expand], dim=-1)
        edge_logits = self.edge_head(fused).squeeze(-1).masked_fill(~edge_mask, -1e4)
        row_no_match_logits = self.row_no_match_head(row_tokens).squeeze(-1).masked_fill(row_padding_mask, -1e4)
        col_newborn_logits = self.col_newborn_head(col_tokens).squeeze(-1).masked_fill(col_padding_mask, -1e4)

        global_edge_mean = masked_mean(encoded_edges.view(batch_size, row_count * col_count, -1), edge_mask.view(batch_size, -1), dim=1)
        global_edge_max = masked_max(encoded_edges.view(batch_size, row_count * col_count, -1), edge_mask.view(batch_size, -1), dim=1)
        global_row_mean = masked_mean(row_tokens, row_present, dim=1)
        global_row_max = masked_max(row_tokens, row_present, dim=1)
        global_col_mean = masked_mean(col_tokens, col_present, dim=1)
        global_col_max = masked_max(col_tokens, col_present, dim=1)
        block_summary = torch.cat(
            [
                global_edge_mean,
                global_edge_max,
                global_row_mean,
                global_row_max,
                global_col_mean,
                global_col_max,
            ],
            dim=-1,
        )
        block_confidence_logits = self.block_confidence_head(block_summary).squeeze(-1)
        return BlockPrimitiveOutput(
            edge_logits=edge_logits,
            row_no_match_logits=row_no_match_logits,
            col_newborn_logits=col_newborn_logits,
            block_confidence_logits=block_confidence_logits,
        )


def default_num_stages() -> int:
    return int(len(STAGE_NAME_TO_ID))
