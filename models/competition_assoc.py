from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


ACTION_LABELS = ("keep", "rerank", "null")
OBSERVED_GROUP_FEATURES = (
    "group_size",
    "candidate_count_total",
    "rank_margin",
    "rank_entropy",
    "track_gap_min",
    "track_gap_mean",
    "track_hist_len_mean",
    "det_score",
)

CANDIDATE_FEATURES = (
    "base_score",
    "refined_score",
    "motion_score",
    "track_gap",
    "track_hist_len",
    "track_rank_frac",
)


def _masked_mean(x: torch.Tensor, valid_mask: torch.Tensor, dim: int) -> torch.Tensor:
    weight = valid_mask.to(dtype=x.dtype)
    denom = weight.sum(dim=dim, keepdim=True).clamp(min=1e-6)
    return (x * weight.unsqueeze(-1)).sum(dim=dim, keepdim=False) / denom


class CompetitionAssociationController(nn.Module):
    """
    Sparse controller for local association conflicts.

    This is intentionally scoped as a bounded decision module:
    - group head predicts whether to keep, rerank, or choose null
    - candidate head only operates inside the current local top-k set
    - continuity head provides an auxiliary short-horizon score
    """

    def __init__(
        self,
        group_dim: int,
        candidate_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.group_dim = int(group_dim)
        self.candidate_dim = int(candidate_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(max(num_heads, 1))

        self.group_proj = nn.Sequential(
            nn.Linear(self.group_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(dropout),
        )
        self.candidate_proj = nn.Sequential(
            nn.Linear(self.candidate_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(dropout),
        )
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.candidate_ln = nn.LayerNorm(self.hidden_dim)
        self.fuse = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(dropout),
        )
        self.action_head = nn.Linear(self.hidden_dim, len(ACTION_LABELS))
        self.candidate_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, 1),
        )
        self.continuity_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )

    def forward(
        self,
        group_features: torch.Tensor,
        candidate_features: torch.Tensor,
        valid_mask: torch.Tensor,
        candidate_bias: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        if group_features.ndim != 2:
            raise ValueError(f"group_features must have shape [B, D], got {tuple(group_features.shape)}")
        if candidate_features.ndim != 3:
            raise ValueError(f"candidate_features must have shape [B, K, D], got {tuple(candidate_features.shape)}")
        if valid_mask.ndim != 2:
            raise ValueError(f"valid_mask must have shape [B, K], got {tuple(valid_mask.shape)}")

        valid_mask = valid_mask.bool()
        group_tokens = self.group_proj(group_features)
        cand_tokens = self.candidate_proj(candidate_features)
        cand_tokens = cand_tokens.masked_fill(~valid_mask.unsqueeze(-1), 0.0)

        key_padding_mask = ~valid_mask
        all_invalid_rows = key_padding_mask.all(dim=1)
        if bool(all_invalid_rows.any()):
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_invalid_rows, 0] = False
            cand_tokens = cand_tokens.clone()
            cand_tokens[all_invalid_rows, 0] = 0.0
        attn_out, _ = self.self_attn(
            cand_tokens,
            cand_tokens,
            cand_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        attn_out = attn_out.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        cand_tokens = self.candidate_ln(cand_tokens + attn_out)
        cand_tokens = cand_tokens.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        pooled = _masked_mean(cand_tokens, valid_mask, dim=1)
        fused = self.fuse(torch.cat([group_tokens, pooled], dim=-1))

        action_logits = self.action_head(fused)
        repeated = fused.unsqueeze(1).expand(-1, cand_tokens.shape[1], -1)
        candidate_logits = self.candidate_head(torch.cat([cand_tokens, repeated], dim=-1)).squeeze(-1)
        if candidate_bias is not None:
            candidate_logits = candidate_logits + candidate_bias
        action_logits = torch.nan_to_num(action_logits, nan=0.0, posinf=20.0, neginf=-20.0)
        candidate_logits = torch.nan_to_num(candidate_logits, nan=0.0, posinf=20.0, neginf=-20.0)
        candidate_logits = candidate_logits.masked_fill(~valid_mask, -1e9)
        continuity_logit = self.continuity_head(fused).squeeze(-1)
        continuity_logit = torch.nan_to_num(continuity_logit, nan=0.0, posinf=20.0, neginf=-20.0)

        return {
            "action_logits": action_logits,
            "candidate_logits": candidate_logits,
            "continuity_logit": continuity_logit,
            "action_prob": torch.softmax(action_logits, dim=-1),
            "candidate_prob": torch.softmax(candidate_logits, dim=-1),
            "continuity_prob": torch.sigmoid(continuity_logit),
        }

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, map_location: str | torch.device = "cpu") -> "CompetitionAssociationController":
        ckpt = torch.load(checkpoint_path, map_location=map_location)
        config = dict(ckpt.get("config", {}))
        model = cls(
            group_dim=int(config.get("group_dim", len(OBSERVED_GROUP_FEATURES))),
            candidate_dim=int(config.get("candidate_dim", len(CANDIDATE_FEATURES))),
            hidden_dim=int(config.get("hidden_dim", 128)),
            dropout=float(config.get("dropout", 0.1)),
        )
        state_dict = ckpt.get("model", ckpt)
        model.load_state_dict(state_dict, strict=True)
        return model


def build_group_feature_tensor(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Canonical group-level features for the first controller baseline.
    """

    feats = [
        batch["group_size"].to(dtype=torch.float32),
        batch["candidate_count_total"].to(dtype=torch.float32),
        batch["rank_margin"].to(dtype=torch.float32),
        batch["rank_entropy"].to(dtype=torch.float32),
        batch["positive_in_topk"].to(dtype=torch.float32),
        batch["group_is_ambiguous"].to(dtype=torch.float32),
        batch["group_is_recoverable"].to(dtype=torch.float32),
        batch["group_is_background"].to(dtype=torch.float32),
        batch["track_gap_min"].to(dtype=torch.float32),
        batch["track_gap_mean"].to(dtype=torch.float32),
        batch["track_hist_len_mean"].to(dtype=torch.float32),
        batch["det_score"].to(dtype=torch.float32),
        batch["future_visible_count"].to(dtype=torch.float32),
        batch["next_same_gt_gap"].to(dtype=torch.float32),
    ]
    return torch.stack(feats, dim=-1)


def build_candidate_feature_tensor(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Canonical candidate-level features for the first controller baseline.
    """

    feats = [
        batch["base_score"].to(dtype=torch.float32),
        batch["refined_score"].to(dtype=torch.float32),
        batch["motion_score"].to(dtype=torch.float32),
        batch["track_gap"].to(dtype=torch.float32),
        batch["track_hist_len"].to(dtype=torch.float32),
        batch["track_rank_frac"].to(dtype=torch.float32),
    ]
    return torch.stack(feats, dim=-1)
