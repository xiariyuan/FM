from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from projects.fgas.fgas.model.block_resolver import STAGE_NAME_TO_ID
from projects.fgas.fgas.model.block_resolver_v3_trackquery import TrackDetCrossLayer, masked_max, masked_mean


MODEL_FAMILY = "fgas_block_matcher"


@dataclass
class BlockMatcherOutput:
    edge_logits: torch.Tensor
    row_no_match_logits: torch.Tensor
    col_newborn_logits: torch.Tensor


class FGASTrueBlockMatcher(nn.Module):
    """
    Block-level matcher trained against whole-block assignment structure.

    Compared with the old primitive family, this model is not optimized around
    row-wise edits or takeover gating. It produces a local assignment energy
    field that is decoded jointly over the whole ambiguity block.
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
        use_base_residual: bool = False,
        base_score_index: int = -1,
        base_logit_scale: float = 0.35,
        edge_aux_indices: list[int] | None = None,
        row_aux_indices: list[int] | None = None,
        col_aux_indices: list[int] | None = None,
        side_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.use_base_residual = bool(use_base_residual)
        self.base_score_index = int(base_score_index)
        self.base_logit_scale = float(base_logit_scale)
        if self.use_base_residual and self.base_score_index < 0:
            raise ValueError("base_score_index must be non-negative when use_base_residual=True")
        edge_aux_indices = sorted({int(v) for v in (edge_aux_indices or [])})
        row_aux_indices = sorted({int(v) for v in (row_aux_indices or [])})
        col_aux_indices = sorted({int(v) for v in (col_aux_indices or [])})
        edge_main_indices = [idx for idx in range(int(input_dim)) if idx not in edge_aux_indices]
        row_main_indices = [idx for idx in range(int(row_context_dim)) if idx not in row_aux_indices]
        col_main_indices = [idx for idx in range(int(col_context_dim)) if idx not in col_aux_indices]
        if len(edge_main_indices) <= 0 or len(row_main_indices) <= 0 or len(col_main_indices) <= 0:
            raise ValueError("main feature split must keep at least one edge/row/col feature")
        self.register_buffer("edge_main_indices", torch.tensor(edge_main_indices, dtype=torch.long), persistent=False)
        self.register_buffer("edge_aux_indices", torch.tensor(edge_aux_indices, dtype=torch.long), persistent=False)
        self.register_buffer("row_main_indices", torch.tensor(row_main_indices, dtype=torch.long), persistent=False)
        self.register_buffer("row_aux_indices", torch.tensor(row_aux_indices, dtype=torch.long), persistent=False)
        self.register_buffer("col_main_indices", torch.tensor(col_main_indices, dtype=torch.long), persistent=False)
        self.register_buffer("col_aux_indices", torch.tensor(col_aux_indices, dtype=torch.long), persistent=False)
        self.use_side_context = bool(edge_aux_indices or row_aux_indices or col_aux_indices)
        self.stage_embedding = nn.Embedding(int(num_stages), int(stage_embed_dim))
        self.edge_encoder = nn.Sequential(
            nn.Linear(len(edge_main_indices) + int(stage_embed_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
        )
        self.row_context_encoder = nn.Sequential(
            nn.Linear(len(row_main_indices) + int(stage_embed_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )
        self.col_context_encoder = nn.Sequential(
            nn.Linear(len(col_main_indices) + int(stage_embed_dim), int(hidden_dim)),
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
        if self.use_side_context:
            init_scale = min(max(float(side_init_scale), 1e-4), 1.0 - 1e-4)
            side_gate_init = torch.logit(torch.tensor(init_scale, dtype=torch.float32))
            self.side_gate_logit = nn.Parameter(side_gate_init.clone())
            self.side_edge_encoder = nn.Sequential(
                nn.Linear(len(edge_aux_indices) + int(stage_embed_dim), int(hidden_dim)),
                nn.ReLU(inplace=True),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
            )
            self.side_row_context_encoder = nn.Sequential(
                nn.Linear(len(row_aux_indices) + int(stage_embed_dim), int(hidden_dim)),
                nn.ReLU(inplace=True),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
            )
            self.side_col_context_encoder = nn.Sequential(
                nn.Linear(len(col_aux_indices) + int(stage_embed_dim), int(hidden_dim)),
                nn.ReLU(inplace=True),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
            )
            self.side_row_edge_proj = nn.Sequential(
                nn.Linear(int(hidden_dim) * 2, int(hidden_dim)),
                nn.ReLU(inplace=True),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
            )
            self.side_col_edge_proj = nn.Sequential(
                nn.Linear(int(hidden_dim) * 2, int(hidden_dim)),
                nn.ReLU(inplace=True),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
            )
            self.side_edge_fuse = nn.Sequential(
                nn.Linear(int(hidden_dim) * 3, int(hidden_dim)),
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
            nn.Linear(int(hidden_dim) * 5, int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), 1),
        )
        self.row_no_match_head = nn.Sequential(
            nn.Linear(int(hidden_dim) * 2, int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), 1),
        )
        self.col_newborn_head = nn.Sequential(
            nn.Linear(int(hidden_dim) * 2, int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(
        self,
        edge_features: torch.Tensor,
        edge_mask: torch.Tensor,
        stage_ids: torch.Tensor,
        row_context: torch.Tensor,
        col_context: torch.Tensor,
    ) -> BlockMatcherOutput:
        batch_size, row_count, col_count, _ = edge_features.shape
        stage_embed = self.stage_embedding(stage_ids)
        stage_edge = stage_embed.view(batch_size, 1, 1, -1).expand(batch_size, row_count, col_count, -1)
        stage_row = stage_embed.view(batch_size, 1, -1).expand(batch_size, row_count, -1)
        stage_col = stage_embed.view(batch_size, 1, -1).expand(batch_size, col_count, -1)

        main_edge_features = edge_features.index_select(dim=-1, index=self.edge_main_indices)
        main_row_context = row_context.index_select(dim=-1, index=self.row_main_indices)
        main_col_context = col_context.index_select(dim=-1, index=self.col_main_indices)
        encoded_edges = self.edge_encoder(torch.cat([main_edge_features, stage_edge], dim=-1))
        row_present = edge_mask.any(dim=2)
        col_present = edge_mask.any(dim=1)
        row_padding_mask = ~row_present
        col_padding_mask = ~col_present

        row_mean = masked_mean(encoded_edges, edge_mask, dim=2)
        row_max = masked_max(encoded_edges, edge_mask, dim=2)
        col_mask = edge_mask.transpose(1, 2)
        col_mean = masked_mean(encoded_edges.transpose(1, 2), col_mask, dim=2)
        col_max = masked_max(encoded_edges.transpose(1, 2), col_mask, dim=2)

        row_tokens = self.row_context_encoder(torch.cat([main_row_context, stage_row], dim=-1)) + self.row_edge_proj(
            torch.cat([row_mean, row_max], dim=-1)
        )
        col_tokens = self.col_context_encoder(torch.cat([main_col_context, stage_col], dim=-1)) + self.col_edge_proj(
            torch.cat([col_mean, col_max], dim=-1)
        )
        if self.use_side_context:
            aux_edge_features = edge_features.index_select(dim=-1, index=self.edge_aux_indices)
            aux_row_context = row_context.index_select(dim=-1, index=self.row_aux_indices)
            aux_col_context = col_context.index_select(dim=-1, index=self.col_aux_indices)
            side_encoded_edges = self.side_edge_encoder(torch.cat([aux_edge_features, stage_edge], dim=-1))
            side_row_mean = masked_mean(side_encoded_edges, edge_mask, dim=2)
            side_row_max = masked_max(side_encoded_edges, edge_mask, dim=2)
            side_col_mean = masked_mean(side_encoded_edges.transpose(1, 2), col_mask, dim=2)
            side_col_max = masked_max(side_encoded_edges.transpose(1, 2), col_mask, dim=2)
            side_scale = torch.sigmoid(self.side_gate_logit)
            row_tokens = row_tokens + side_scale * (
                self.side_row_context_encoder(torch.cat([aux_row_context, stage_row], dim=-1))
                + self.side_row_edge_proj(torch.cat([side_row_mean, side_row_max], dim=-1))
            )
            col_tokens = col_tokens + side_scale * (
                self.side_col_context_encoder(torch.cat([aux_col_context, stage_col], dim=-1))
                + self.side_col_edge_proj(torch.cat([side_col_mean, side_col_max], dim=-1))
            )
            encoded_edges = encoded_edges + side_scale * self.side_edge_fuse(
                torch.cat([encoded_edges, side_encoded_edges, torch.abs(encoded_edges - side_encoded_edges)], dim=-1)
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
        fused = torch.cat(
            [
                encoded_edges,
                row_expand,
                col_expand,
                row_expand * col_expand,
                torch.abs(row_expand - col_expand),
            ],
            dim=-1,
        )
        edge_logits = self.edge_head(fused).squeeze(-1)
        if self.use_base_residual:
            base_scores = edge_features[..., self.base_score_index].clamp(min=1e-4, max=1.0 - 1e-4)
            edge_logits = edge_logits + float(self.base_logit_scale) * torch.logit(base_scores)
        edge_logits = edge_logits.masked_fill(~edge_mask, -1e4)

        row_summary = torch.cat([row_tokens, row_mean], dim=-1)
        col_summary = torch.cat([col_tokens, col_mean], dim=-1)
        row_no_match_logits = self.row_no_match_head(row_summary).squeeze(-1).masked_fill(row_padding_mask, -1e4)
        col_newborn_logits = self.col_newborn_head(col_summary).squeeze(-1).masked_fill(col_padding_mask, -1e4)
        return BlockMatcherOutput(
            edge_logits=edge_logits,
            row_no_match_logits=row_no_match_logits,
            col_newborn_logits=col_newborn_logits,
        )


def default_num_stages() -> int:
    return int(len(STAGE_NAME_TO_ID))
