from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
from torch import nn

from models.local_conflict_commit import (
    CLUSTER_FEATURE_NAMES,
    DET_FEATURE_NAMES,
    EDGE_FEATURE_NAMES,
    TRACK_FEATURE_NAMES,
    LocalConflictCommitRefiner,
)


ACTION_NAMES = ("rewrite", "defer", "reject")
MODEL_ARCH_LEGACY = "legacy_pool_v1"
MODEL_ARCH_SET_SLOT = "set_slot_v1"
MODEL_ARCH_ROUTED_MOE = "routed_moe_v1"
MODEL_ARCH_HIER_ROUTE = "hier_route_v1"


def _pool_mean_max(values: torch.Tensor) -> torch.Tensor:
    if values.ndim == 1:
        values = values.view(1, -1)
    if values.numel() == 0:
        return values.new_zeros((int(values.shape[-1]) * 2,))
    return torch.cat([values.mean(dim=0), values.max(dim=0).values], dim=-1)


def _row_entropy(dense_logits: torch.Tensor) -> torch.Tensor:
    if dense_logits.ndim != 2 or dense_logits.numel() == 0:
        return dense_logits.new_zeros((0,))
    probs = torch.softmax(dense_logits - dense_logits.max(dim=-1, keepdim=True).values, dim=-1)
    return -(probs * torch.log(probs.clamp(min=1e-8))).sum(dim=-1)


def _assignment_summary(
    *,
    edge_logits: torch.Tensor,
    defer_logits: torch.Tensor,
    dense_logits: torch.Tensor,
    num_tracks: int,
) -> torch.Tensor:
    if dense_logits.ndim != 2 or dense_logits.shape[0] == 0:
        return dense_logits.new_zeros((12,))

    num_dets = int(dense_logits.shape[0])
    if num_tracks > 0:
        track_logits = dense_logits[:, : int(num_tracks)]
        topk = torch.topk(track_logits, k=min(2, int(num_tracks)), dim=-1, sorted=True).values
        row_top1 = topk[:, 0]
        row_top2 = topk[:, 1] if topk.shape[1] > 1 else row_top1.new_zeros(row_top1.shape)
    else:
        row_top1 = dense_logits.new_zeros((num_dets,))
        row_top2 = dense_logits.new_zeros((num_dets,))
    row_margin = row_top1 - row_top2
    row_entropy = _row_entropy(dense_logits)

    edge_logits = edge_logits.view(-1)
    defer_logits = defer_logits.view(-1)
    summary = torch.tensor(
        [
            float(row_top1.mean().item()) if row_top1.numel() > 0 else 0.0,
            float(row_top1.max().item()) if row_top1.numel() > 0 else 0.0,
            float(row_top2.mean().item()) if row_top2.numel() > 0 else 0.0,
            float(row_top2.max().item()) if row_top2.numel() > 0 else 0.0,
            float(row_margin.mean().item()) if row_margin.numel() > 0 else 0.0,
            float(row_margin.max().item()) if row_margin.numel() > 0 else 0.0,
            float(row_entropy.mean().item()) if row_entropy.numel() > 0 else 0.0,
            float(row_entropy.max().item()) if row_entropy.numel() > 0 else 0.0,
            float(edge_logits.mean().item()) if edge_logits.numel() > 0 else 0.0,
            float(edge_logits.max().item()) if edge_logits.numel() > 0 else 0.0,
            float(defer_logits.mean().item()) if defer_logits.numel() > 0 else 0.0,
            float(defer_logits.max().item()) if defer_logits.numel() > 0 else 0.0,
        ],
        dtype=torch.float32,
        device=dense_logits.device,
    )
    return summary


@dataclass
class GraphAssocCommitPolicyConfig:
    det_dim: int = len(DET_FEATURE_NAMES)
    track_dim: int = len(TRACK_FEATURE_NAMES)
    edge_dim: int = len(EDGE_FEATURE_NAMES)
    cluster_dim: int = len(CLUSTER_FEATURE_NAMES)
    hidden_dim: int = 128
    policy_hidden_dim: int = 128
    dropout: float = 0.1


class GraphAssocCommitPolicy(nn.Module):
    def __init__(
        self,
        det_dim: int = len(DET_FEATURE_NAMES),
        track_dim: int = len(TRACK_FEATURE_NAMES),
        edge_dim: int = len(EDGE_FEATURE_NAMES),
        cluster_dim: int = len(CLUSTER_FEATURE_NAMES),
        hidden_dim: int = 128,
        policy_hidden_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.det_dim = int(det_dim)
        self.track_dim = int(track_dim)
        self.edge_dim = int(edge_dim)
        self.cluster_dim = int(cluster_dim)
        self.hidden_dim = int(hidden_dim)
        self.policy_hidden_dim = int(policy_hidden_dim if policy_hidden_dim is not None else hidden_dim)
        self.dropout = float(dropout)
        self.model_type = "action_policy"

        self.backbone = LocalConflictCommitRefiner(
            det_dim=self.det_dim,
            track_dim=self.track_dim,
            edge_dim=self.edge_dim,
            cluster_dim=self.cluster_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        )

        policy_input_dim = self.hidden_dim * 15 + self.cluster_dim + 12
        self.policy_norm = nn.LayerNorm(policy_input_dim)
        self.policy_backbone = nn.Sequential(
            nn.Linear(policy_input_dim, self.policy_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.policy_hidden_dim, self.policy_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.action_head = nn.Linear(self.policy_hidden_dim, len(ACTION_NAMES))
        self.gain_head = nn.Linear(self.policy_hidden_dim, 1)

    def _build_policy_features(
        self,
        *,
        det_hidden: torch.Tensor,
        track_hidden: torch.Tensor,
        edge_hidden: torch.Tensor,
        row_ctx: torch.Tensor,
        col_ctx: torch.Tensor,
        cluster_hidden: torch.Tensor,
        cluster_features: torch.Tensor,
        edge_logits: torch.Tensor,
        defer_logits: torch.Tensor,
        dense_logits: torch.Tensor,
    ) -> torch.Tensor:
        hidden_parts = [
            _pool_mean_max(det_hidden),
            _pool_mean_max(track_hidden),
            _pool_mean_max(edge_hidden),
            _pool_mean_max(row_ctx),
            _pool_mean_max(col_ctx),
            cluster_hidden.view(-1).to(dtype=torch.float32),
            cluster_features.view(-1).to(dtype=torch.float32),
            _assignment_summary(
                edge_logits=edge_logits,
                defer_logits=defer_logits,
                dense_logits=dense_logits,
                num_tracks=int(dense_logits.shape[-1] - 1),
            ),
        ]
        return torch.cat(hidden_parts, dim=-1)

    def forward(
        self,
        det_features: torch.Tensor,
        track_features: torch.Tensor,
        edge_features: torch.Tensor,
        edge_det_index: torch.Tensor,
        edge_track_index: torch.Tensor,
        cluster_features: torch.Tensor,
        *,
        return_context: bool = False,
    ) -> Dict[str, torch.Tensor]:
        backbone_outputs = self.backbone(
            det_features=det_features,
            track_features=track_features,
            edge_features=edge_features,
            edge_det_index=edge_det_index,
            edge_track_index=edge_track_index,
            cluster_features=cluster_features,
            return_context=True,
        )

        edge_logits = backbone_outputs["edge_logits"]
        defer_logits = backbone_outputs["defer_logits"]
        dense_logits = LocalConflictCommitRefiner.build_dense_assignment_logits(
            num_detections=int(det_features.shape[0]),
            num_tracks=int(track_features.shape[0]),
            edge_logits=edge_logits,
            edge_det_index=edge_det_index,
            edge_track_index=edge_track_index,
            defer_logits=defer_logits,
        )
        policy_features = self._build_policy_features(
            det_hidden=backbone_outputs["det_hidden"],
            track_hidden=backbone_outputs["track_hidden"],
            edge_hidden=backbone_outputs["edge_hidden"],
            row_ctx=backbone_outputs["row_ctx"],
            col_ctx=backbone_outputs["col_ctx"],
            cluster_hidden=backbone_outputs["cluster_hidden"],
            cluster_features=cluster_features.to(dtype=torch.float32).view(-1),
            edge_logits=edge_logits,
            defer_logits=defer_logits,
            dense_logits=dense_logits,
        )
        policy_hidden = self.policy_backbone(self.policy_norm(policy_features))
        action_logits = self.action_head(policy_hidden)
        action_probs = torch.softmax(action_logits, dim=-1)
        gain_raw = self.gain_head(policy_hidden).squeeze(-1)
        gain_pred = 2.0 * torch.tanh(gain_raw / 2.0)
        action_margin = action_probs[..., 0] - action_probs[..., 2]
        policy_score = gain_pred + 0.5 * (action_probs[..., 0] - action_probs[..., 2])

        outputs: Dict[str, torch.Tensor] = {
            "edge_logits": edge_logits,
            "defer_logits": defer_logits,
            "action_logits": action_logits,
            "action_probs": action_probs,
            "gain_raw": gain_raw,
            "gain_pred": gain_pred,
            "action_margin": action_margin,
            "policy_score": policy_score,
            "decision_score": policy_score,
        }
        if return_context:
            outputs.update(
                {
                    "policy_features": policy_features,
                    "policy_hidden": policy_hidden,
                    "dense_logits": dense_logits,
                    "det_hidden": backbone_outputs["det_hidden"],
                    "track_hidden": backbone_outputs["track_hidden"],
                    "edge_hidden": backbone_outputs["edge_hidden"],
                    "cluster_hidden": backbone_outputs["cluster_hidden"],
                    "row_ctx": backbone_outputs["row_ctx"],
                    "col_ctx": backbone_outputs["col_ctx"],
                }
            )
        return outputs

    def checkpoint_payload(self, **extra: Any) -> Dict[str, Any]:
        payload = {
            "model_type": "action_policy",
            "model_arch": MODEL_ARCH_LEGACY,
            "model_state": self.state_dict(),
            "model_kwargs": {
                "det_dim": self.det_dim,
                "track_dim": self.track_dim,
                "edge_dim": self.edge_dim,
                "cluster_dim": self.cluster_dim,
                "hidden_dim": self.hidden_dim,
                "policy_hidden_dim": self.policy_hidden_dim,
                "dropout": self.dropout,
            },
            "feature_names": {
                "det": list(DET_FEATURE_NAMES),
                "track": list(TRACK_FEATURE_NAMES),
                "edge": list(EDGE_FEATURE_NAMES),
                "cluster": list(CLUSTER_FEATURE_NAMES),
            },
            "action_names": list(ACTION_NAMES),
        }
        payload.update(extra)
        return payload

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        *,
        map_location: str | torch.device | None = None,
    ) -> "GraphAssocCommitPolicy":
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        model_arch = str(checkpoint.get("model_arch", MODEL_ARCH_LEGACY) or MODEL_ARCH_LEGACY)
        if model_arch == MODEL_ARCH_ROUTED_MOE:
            return GraphAssocCommitRoutedSetSlotPolicy.from_checkpoint(checkpoint_path, map_location=map_location)
        if model_arch == MODEL_ARCH_SET_SLOT:
            return GraphAssocCommitSetSlotPolicy.from_checkpoint(checkpoint_path, map_location=map_location)
        if model_arch == MODEL_ARCH_HIER_ROUTE:
            return GraphAssocCommitHierarchicalRoutePolicy.from_checkpoint(checkpoint_path, map_location=map_location)
        model_kwargs = dict(checkpoint.get("model_kwargs", {}))
        model = cls(**model_kwargs)
        state = checkpoint.get("model_state", checkpoint)
        model.load_state_dict(state, strict=True)
        model.eval()
        return model


class _SetSlotEncoderBlock(nn.Module):
    def __init__(self, token_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(int(token_dim), int(num_heads), dropout=float(dropout), batch_first=True)
        self.norm1 = nn.LayerNorm(int(token_dim))
        self.ff = nn.Sequential(
            nn.Linear(int(token_dim), int(token_dim) * 4),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(token_dim) * 4, int(token_dim)),
            nn.Dropout(float(dropout)),
        )
        self.norm2 = nn.LayerNorm(int(token_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class GraphAssocCommitSetSlotPolicy(nn.Module):
    def __init__(
        self,
        det_dim: int = len(DET_FEATURE_NAMES),
        track_dim: int = len(TRACK_FEATURE_NAMES),
        edge_dim: int = len(EDGE_FEATURE_NAMES),
        cluster_dim: int = len(CLUSTER_FEATURE_NAMES),
        hidden_dim: int = 128,
        policy_hidden_dim: int | None = None,
        dropout: float = 0.1,
        token_dim: int | None = None,
        num_slots: int = 4,
        num_encoder_layers: int = 2,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.det_dim = int(det_dim)
        self.track_dim = int(track_dim)
        self.edge_dim = int(edge_dim)
        self.cluster_dim = int(cluster_dim)
        self.hidden_dim = int(hidden_dim)
        self.policy_hidden_dim = int(policy_hidden_dim if policy_hidden_dim is not None else hidden_dim)
        self.token_dim = int(token_dim if token_dim is not None else hidden_dim)
        self.dropout = float(dropout)
        self.num_slots = int(num_slots)
        self.num_encoder_layers = int(num_encoder_layers)
        self.num_heads = int(num_heads)
        self.model_type = "action_policy"

        self.backbone = LocalConflictCommitRefiner(
            det_dim=self.det_dim,
            track_dim=self.track_dim,
            edge_dim=self.edge_dim,
            cluster_dim=self.cluster_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        )

        self.det_token_proj = nn.Sequential(
            nn.Linear(self.hidden_dim + self.hidden_dim * 2 + self.det_dim, self.token_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.token_dim, self.token_dim),
        )
        self.track_token_proj = nn.Sequential(
            nn.Linear(self.hidden_dim + self.hidden_dim * 2 + self.track_dim, self.token_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.token_dim, self.token_dim),
        )
        self.edge_token_proj = nn.Sequential(
            nn.Linear(self.hidden_dim + self.edge_dim + 1, self.token_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.token_dim, self.token_dim),
        )
        self.cluster_token_proj = nn.Sequential(
            nn.Linear(self.hidden_dim + self.cluster_dim, self.token_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.token_dim, self.token_dim),
        )
        self.type_embed = nn.Embedding(4, self.token_dim)
        self.token_norm = nn.LayerNorm(self.token_dim)
        self.slot_tokens = nn.Parameter(torch.randn(self.num_slots, self.token_dim) * 0.02)

        self.encoder_blocks = nn.ModuleList(
            [_SetSlotEncoderBlock(self.token_dim, self.num_heads, self.dropout) for _ in range(self.num_encoder_layers)]
        )
        self.action_queries = nn.Parameter(torch.randn(len(ACTION_NAMES), self.token_dim) * 0.02)
        self.action_cross_attn = nn.MultiheadAttention(self.token_dim, self.num_heads, dropout=self.dropout, batch_first=True)
        self.action_query_norm = nn.LayerNorm(self.token_dim)
        self.action_query_bias = nn.Sequential(
            nn.Linear(self.summary_dim, self.token_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.token_dim, self.token_dim),
        )

        policy_input_dim = self.summary_dim + len(ACTION_NAMES) * self.token_dim
        self.policy_norm = nn.LayerNorm(policy_input_dim)
        self.policy_backbone = nn.Sequential(
            nn.Linear(policy_input_dim, self.policy_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.policy_hidden_dim, self.policy_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.action_head = nn.Linear(self.policy_hidden_dim, len(ACTION_NAMES))
        self.gain_head = nn.Linear(self.policy_hidden_dim, 1)

    @property
    def summary_dim(self) -> int:
        return int(self.token_dim * 4 + self.hidden_dim + self.cluster_dim + 12)

    def _project_tokens(
        self,
        *,
        det_features: torch.Tensor,
        track_features: torch.Tensor,
        edge_features: torch.Tensor,
        edge_logits: torch.Tensor,
        det_hidden: torch.Tensor,
        track_hidden: torch.Tensor,
        edge_hidden: torch.Tensor,
        row_ctx: torch.Tensor,
        col_ctx: torch.Tensor,
        cluster_hidden: torch.Tensor,
        cluster_features: torch.Tensor,
    ) -> torch.Tensor:
        token_parts: list[torch.Tensor] = []
        type_ids: list[torch.Tensor] = []

        if det_hidden.numel() > 0:
            det_input = torch.cat([det_hidden, row_ctx, det_features], dim=-1)
            token_parts.append(self.det_token_proj(det_input))
            type_ids.append(torch.zeros((int(det_input.shape[0]),), dtype=torch.long, device=det_input.device))

        if track_hidden.numel() > 0:
            track_input = torch.cat([track_hidden, col_ctx, track_features], dim=-1)
            token_parts.append(self.track_token_proj(track_input))
            type_ids.append(torch.ones((int(track_input.shape[0]),), dtype=torch.long, device=track_input.device))

        if edge_hidden.numel() > 0:
            edge_input = torch.cat([edge_hidden, edge_features, edge_logits.view(-1, 1)], dim=-1)
            token_parts.append(self.edge_token_proj(edge_input))
            type_ids.append(torch.full((int(edge_input.shape[0]),), 2, dtype=torch.long, device=edge_input.device))

        cluster_input = torch.cat([cluster_hidden.view(1, -1), cluster_features.view(1, -1)], dim=-1)
        token_parts.append(self.cluster_token_proj(cluster_input))
        type_ids.append(torch.full((1,), 3, dtype=torch.long, device=cluster_input.device))

        tokens = torch.cat(token_parts, dim=0) if token_parts else cluster_hidden.new_zeros((0, self.token_dim))
        type_bias = self.type_embed(torch.cat(type_ids, dim=0)) if type_ids else tokens.new_zeros(tokens.shape)
        return self.token_norm(tokens + type_bias)

    def _encode_set(self, tokens: torch.Tensor) -> torch.Tensor:
        slot_tokens = self.token_norm(self.slot_tokens.unsqueeze(0))
        seq = torch.cat([slot_tokens, tokens.unsqueeze(0)], dim=1)
        for block in self.encoder_blocks:
            seq = block(seq)
        return seq

    def _build_summary(
        self,
        *,
        slot_tokens: torch.Tensor,
        token_tokens: torch.Tensor,
        cluster_hidden: torch.Tensor,
        cluster_features: torch.Tensor,
        edge_logits: torch.Tensor,
        defer_logits: torch.Tensor,
        dense_logits: torch.Tensor,
        num_tracks: int,
    ) -> torch.Tensor:
        slot_pool = _pool_mean_max(slot_tokens)
        token_pool = _pool_mean_max(token_tokens) if token_tokens.numel() > 0 else token_tokens.new_zeros((self.token_dim * 2,))
        assignment_summary = _assignment_summary(
            edge_logits=edge_logits,
            defer_logits=defer_logits,
            dense_logits=dense_logits,
            num_tracks=num_tracks,
        )
        return torch.cat([slot_pool, token_pool, cluster_hidden.view(-1), cluster_features.view(-1), assignment_summary], dim=-1)

    def forward(
        self,
        det_features: torch.Tensor,
        track_features: torch.Tensor,
        edge_features: torch.Tensor,
        edge_det_index: torch.Tensor,
        edge_track_index: torch.Tensor,
        cluster_features: torch.Tensor,
        *,
        return_context: bool = False,
    ) -> Dict[str, torch.Tensor]:
        backbone_outputs = self.backbone(
            det_features=det_features,
            track_features=track_features,
            edge_features=edge_features,
            edge_det_index=edge_det_index,
            edge_track_index=edge_track_index,
            cluster_features=cluster_features,
            return_context=True,
        )

        edge_logits = backbone_outputs["edge_logits"]
        defer_logits = backbone_outputs["defer_logits"]
        dense_logits = LocalConflictCommitRefiner.build_dense_assignment_logits(
            num_detections=int(det_features.shape[0]),
            num_tracks=int(track_features.shape[0]),
            edge_logits=edge_logits,
            edge_det_index=edge_det_index,
            edge_track_index=edge_track_index,
            defer_logits=defer_logits,
        )
        tokens = self._project_tokens(
            det_features=det_features.to(dtype=torch.float32),
            track_features=track_features.to(dtype=torch.float32),
            edge_features=edge_features.to(dtype=torch.float32),
            edge_logits=edge_logits,
            det_hidden=backbone_outputs["det_hidden"],
            track_hidden=backbone_outputs["track_hidden"],
            edge_hidden=backbone_outputs["edge_hidden"],
            row_ctx=backbone_outputs["row_ctx"],
            col_ctx=backbone_outputs["col_ctx"],
            cluster_hidden=backbone_outputs["cluster_hidden"],
            cluster_features=cluster_features.to(dtype=torch.float32).view(-1),
        )
        seq = self._encode_set(tokens)
        slot_tokens = seq[:, : self.num_slots, :]
        token_tokens = seq[:, self.num_slots :, :]

        summary = self._build_summary(
            slot_tokens=slot_tokens.squeeze(0),
            token_tokens=token_tokens.squeeze(0),
            cluster_hidden=backbone_outputs["cluster_hidden"],
            cluster_features=cluster_features.to(dtype=torch.float32).view(-1),
            edge_logits=edge_logits,
            defer_logits=defer_logits,
            dense_logits=dense_logits,
            num_tracks=int(track_features.shape[0]),
        )
        action_query_bias = self.action_query_bias(summary).view(1, 1, -1)
        action_queries = self.action_queries.unsqueeze(0) + action_query_bias
        action_ctx, _ = self.action_cross_attn(action_queries, seq, seq, need_weights=False)
        action_ctx = self.action_query_norm(action_ctx + action_queries)

        policy_context = torch.cat([summary, action_ctx.reshape(-1)], dim=-1)
        policy_hidden = self.policy_backbone(self.policy_norm(policy_context))
        action_logits = self.action_head(policy_hidden)
        action_probs = torch.softmax(action_logits, dim=-1)
        gain_raw = self.gain_head(policy_hidden).squeeze(-1)
        gain_pred = 2.0 * torch.tanh(gain_raw / 2.0)
        action_margin = action_probs[..., 0] - action_probs[..., 2]
        policy_score = gain_pred + 0.5 * (action_probs[..., 0] - action_probs[..., 2])

        outputs: Dict[str, torch.Tensor] = {
            "edge_logits": edge_logits,
            "defer_logits": defer_logits,
            "action_logits": action_logits,
            "action_probs": action_probs,
            "gain_raw": gain_raw,
            "gain_pred": gain_pred,
            "action_margin": action_margin,
            "policy_score": policy_score,
            "decision_score": policy_score,
        }
        if return_context:
            outputs.update(
                {
                    "policy_features": policy_context,
                    "policy_hidden": policy_hidden,
                    "dense_logits": dense_logits,
                    "det_hidden": backbone_outputs["det_hidden"],
                    "track_hidden": backbone_outputs["track_hidden"],
                    "edge_hidden": backbone_outputs["edge_hidden"],
                    "cluster_hidden": backbone_outputs["cluster_hidden"],
                    "row_ctx": backbone_outputs["row_ctx"],
                    "col_ctx": backbone_outputs["col_ctx"],
                    "slot_tokens": slot_tokens.squeeze(0),
                    "token_tokens": token_tokens.squeeze(0),
                    "action_ctx": action_ctx.squeeze(0),
                }
            )
        return outputs

    def checkpoint_payload(self, **extra: Any) -> Dict[str, Any]:
        payload = {
            "model_type": "action_policy",
            "model_arch": MODEL_ARCH_SET_SLOT,
            "model_state": self.state_dict(),
            "model_kwargs": {
                "det_dim": self.det_dim,
                "track_dim": self.track_dim,
                "edge_dim": self.edge_dim,
                "cluster_dim": self.cluster_dim,
                "hidden_dim": self.hidden_dim,
                "policy_hidden_dim": self.policy_hidden_dim,
                "dropout": self.dropout,
                "token_dim": self.token_dim,
                "num_slots": self.num_slots,
                "num_encoder_layers": self.num_encoder_layers,
                "num_heads": self.num_heads,
            },
            "feature_names": {
                "det": list(DET_FEATURE_NAMES),
                "track": list(TRACK_FEATURE_NAMES),
                "edge": list(EDGE_FEATURE_NAMES),
                "cluster": list(CLUSTER_FEATURE_NAMES),
            },
            "action_names": list(ACTION_NAMES),
        }
        payload.update(extra)
        return payload

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        *,
        map_location: str | torch.device | None = None,
    ) -> "GraphAssocCommitSetSlotPolicy":
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        model_kwargs = dict(checkpoint.get("model_kwargs", {}))
        model = cls(**model_kwargs)
        state = checkpoint.get("model_state", checkpoint)
        model.load_state_dict(state, strict=False)
        model.eval()
        return model


class GraphAssocCommitRoutedSetSlotPolicy(GraphAssocCommitSetSlotPolicy):
    def __init__(
        self,
        det_dim: int = len(DET_FEATURE_NAMES),
        track_dim: int = len(TRACK_FEATURE_NAMES),
        edge_dim: int = len(EDGE_FEATURE_NAMES),
        cluster_dim: int = len(CLUSTER_FEATURE_NAMES),
        hidden_dim: int = 128,
        policy_hidden_dim: int | None = None,
        dropout: float = 0.1,
        token_dim: int | None = None,
        num_slots: int = 4,
        num_encoder_layers: int = 2,
        num_heads: int = 4,
        num_experts: int = 3,
        router_hidden_dim: int | None = None,
        router_temperature: float = 1.0,
        policy_score_mode: str = "learned_selection",
    ) -> None:
        if int(num_experts) != 3:
            raise ValueError("GraphAssocCommitRoutedSetSlotPolicy currently expects num_experts=3")
        super().__init__(
            det_dim=det_dim,
            track_dim=track_dim,
            edge_dim=edge_dim,
            cluster_dim=cluster_dim,
            hidden_dim=hidden_dim,
            policy_hidden_dim=policy_hidden_dim,
            dropout=dropout,
            token_dim=token_dim,
            num_slots=num_slots,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
        )
        self.num_experts = int(num_experts)
        self.router_hidden_dim = int(router_hidden_dim if router_hidden_dim is not None else self.policy_hidden_dim)
        self.router_temperature = float(router_temperature if float(router_temperature) > 0.0 else 1.0)
        requested_policy_score_mode = str(policy_score_mode or "learned_selection").strip().lower()
        if requested_policy_score_mode not in {
            "learned_selection",
            "legacy_mixture",
            "residual_selection",
            "calibrated_residual",
            "gated_blend",
        }:
            requested_policy_score_mode = "learned_selection"
        self.policy_score_mode = requested_policy_score_mode
        self.model_arch = MODEL_ARCH_ROUTED_MOE

        router_input_dim = self.summary_dim + self.policy_hidden_dim
        self.router_norm = nn.LayerNorm(router_input_dim)
        self.router_backbone = nn.Sequential(
            nn.Linear(router_input_dim, self.router_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.router_hidden_dim, self.router_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.router_head = nn.Linear(self.router_hidden_dim, self.num_experts)

        expert_input_dims = [
            self.policy_hidden_dim + self.token_dim * 2 + 12,
            self.policy_hidden_dim + self.token_dim * 2 + self.hidden_dim + self.cluster_dim,
            self.policy_hidden_dim + self.token_dim * 3 + self.hidden_dim + self.cluster_dim + 12,
        ]
        self.expert_input_dims = list(expert_input_dims)
        self.expert_norms = nn.ModuleList([nn.LayerNorm(dim) for dim in expert_input_dims])
        self.expert_backbones = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, self.policy_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.policy_hidden_dim, self.policy_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                )
                for dim in expert_input_dims
            ]
        )
        self.expert_action_heads = nn.ModuleList([nn.Linear(self.policy_hidden_dim, len(ACTION_NAMES)) for _ in range(self.num_experts)])
        self.expert_gain_heads = nn.ModuleList([nn.Linear(self.policy_hidden_dim, 1) for _ in range(self.num_experts)])
        expert_bias = torch.tensor(
            [
                [0.30, -0.05, -0.25],
                [-0.05, 0.30, -0.05],
                [-0.25, -0.05, 0.30],
            ],
            dtype=torch.float32,
        )
        self.expert_action_bias = nn.Parameter(expert_bias[: self.num_experts].clone())

        self.policy_context_dim = self.summary_dim + len(ACTION_NAMES) * self.token_dim
        self.selection_input_dim = self.policy_context_dim + self.policy_hidden_dim + self.router_hidden_dim + 11
        self.selection_norm = nn.LayerNorm(self.selection_input_dim)
        self.selection_backbone = nn.Sequential(
            nn.Linear(self.selection_input_dim, self.policy_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.policy_hidden_dim, self.policy_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.selection_head = nn.Linear(self.policy_hidden_dim, 1)
        self.calibration_head = nn.Linear(self.policy_hidden_dim, 1)
        self.blend_gate_input_dim = self.policy_hidden_dim + self.router_hidden_dim + 3
        self.blend_gate_norm = nn.LayerNorm(self.blend_gate_input_dim)
        self.blend_gate_backbone = nn.Sequential(
            nn.Linear(self.blend_gate_input_dim, self.policy_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.policy_hidden_dim, self.policy_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.blend_gate_head = nn.Linear(self.policy_hidden_dim, 1)

    @staticmethod
    def _extract_views(
        *,
        slot_tokens: torch.Tensor,
        token_tokens: torch.Tensor,
        cluster_hidden: torch.Tensor,
        cluster_features: torch.Tensor,
        edge_logits: torch.Tensor,
        defer_logits: torch.Tensor,
        dense_logits: torch.Tensor,
        num_tracks: int,
    ) -> Dict[str, torch.Tensor]:
        slot_pool = _pool_mean_max(slot_tokens)
        token_pool = _pool_mean_max(token_tokens) if token_tokens.numel() > 0 else token_tokens.new_zeros((slot_tokens.shape[-1] * 2,))
        assignment_summary = _assignment_summary(
            edge_logits=edge_logits,
            defer_logits=defer_logits,
            dense_logits=dense_logits,
            num_tracks=num_tracks,
        )
        cluster_core = torch.cat([cluster_hidden.view(-1).to(dtype=torch.float32), cluster_features.view(-1).to(dtype=torch.float32)], dim=-1)
        summary = torch.cat([slot_pool, token_pool, cluster_core, assignment_summary], dim=-1)
        return {
            "slot_pool": slot_pool,
            "token_pool": token_pool,
            "cluster_core": cluster_core,
            "assignment_summary": assignment_summary,
            "summary": summary,
        }

    def _build_expert_input(
        self,
        *,
        expert_index: int,
        shared_hidden: torch.Tensor,
        summary: torch.Tensor,
        slot_pool: torch.Tensor,
        token_pool: torch.Tensor,
        cluster_core: torch.Tensor,
        assignment_summary: torch.Tensor,
        action_ctx: torch.Tensor,
    ) -> torch.Tensor:
        if int(expert_index) == 0:
            base = torch.cat([shared_hidden, slot_pool, assignment_summary], dim=-1)
        elif int(expert_index) == 1:
            base = torch.cat([shared_hidden, token_pool, cluster_core], dim=-1)
        else:
            base = torch.cat([shared_hidden, action_ctx.reshape(-1), cluster_core, assignment_summary], dim=-1)
        if base.numel() != int(self.expert_input_dims[int(expert_index)]):
            raise RuntimeError(
                f"expert input size mismatch for expert {int(expert_index)}: "
                f"expected {int(self.expert_input_dims[int(expert_index)])}, got {int(base.numel())}"
            )
        return base

    def _build_selection_input(
        self,
        *,
        policy_context: torch.Tensor,
        policy_hidden: torch.Tensor,
        action_probs: torch.Tensor,
        gain_pred: torch.Tensor,
        action_margin: torch.Tensor,
        legacy_policy_score: torch.Tensor,
        router_hidden: torch.Tensor,
        router_probs: torch.Tensor,
        router_margin: torch.Tensor,
        router_entropy: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            [
                policy_context.reshape(-1).to(dtype=torch.float32),
                policy_hidden.reshape(-1).to(dtype=torch.float32),
                action_probs.reshape(-1).to(dtype=torch.float32),
                gain_pred.reshape(-1).to(dtype=torch.float32),
                action_margin.reshape(-1).to(dtype=torch.float32),
                legacy_policy_score.reshape(-1).to(dtype=torch.float32),
                router_hidden.reshape(-1).to(dtype=torch.float32),
                router_probs.reshape(-1).to(dtype=torch.float32),
                router_margin.reshape(-1).to(dtype=torch.float32),
                router_entropy.reshape(-1).to(dtype=torch.float32),
            ],
            dim=-1,
        )

    def forward(
        self,
        det_features: torch.Tensor,
        track_features: torch.Tensor,
        edge_features: torch.Tensor,
        edge_det_index: torch.Tensor,
        edge_track_index: torch.Tensor,
        cluster_features: torch.Tensor,
        *,
        return_context: bool = False,
    ) -> Dict[str, torch.Tensor]:
        backbone_outputs = self.backbone(
            det_features=det_features,
            track_features=track_features,
            edge_features=edge_features,
            edge_det_index=edge_det_index,
            edge_track_index=edge_track_index,
            cluster_features=cluster_features,
            return_context=True,
        )

        edge_logits = backbone_outputs["edge_logits"]
        defer_logits = backbone_outputs["defer_logits"]
        dense_logits = LocalConflictCommitRefiner.build_dense_assignment_logits(
            num_detections=int(det_features.shape[0]),
            num_tracks=int(track_features.shape[0]),
            edge_logits=edge_logits,
            edge_det_index=edge_det_index,
            edge_track_index=edge_track_index,
            defer_logits=defer_logits,
        )
        tokens = self._project_tokens(
            det_features=det_features.to(dtype=torch.float32),
            track_features=track_features.to(dtype=torch.float32),
            edge_features=edge_features.to(dtype=torch.float32),
            edge_logits=edge_logits,
            det_hidden=backbone_outputs["det_hidden"],
            track_hidden=backbone_outputs["track_hidden"],
            edge_hidden=backbone_outputs["edge_hidden"],
            row_ctx=backbone_outputs["row_ctx"],
            col_ctx=backbone_outputs["col_ctx"],
            cluster_hidden=backbone_outputs["cluster_hidden"],
            cluster_features=cluster_features.to(dtype=torch.float32).view(-1),
        )
        seq = self._encode_set(tokens)
        slot_tokens = seq[:, : self.num_slots, :]
        token_tokens = seq[:, self.num_slots :, :]
        views = self._extract_views(
            slot_tokens=slot_tokens.squeeze(0),
            token_tokens=token_tokens.squeeze(0),
            cluster_hidden=backbone_outputs["cluster_hidden"],
            cluster_features=cluster_features.to(dtype=torch.float32).view(-1),
            edge_logits=edge_logits,
            defer_logits=defer_logits,
            dense_logits=dense_logits,
            num_tracks=int(track_features.shape[0]),
        )

        action_query_bias = self.action_query_bias(views["summary"]).view(1, 1, -1)
        action_queries = self.action_queries.unsqueeze(0) + action_query_bias
        action_ctx, _ = self.action_cross_attn(action_queries, seq, seq, need_weights=False)
        action_ctx = self.action_query_norm(action_ctx + action_queries)

        policy_context = torch.cat([views["summary"], action_ctx.reshape(-1)], dim=-1)
        shared_hidden = self.policy_backbone(self.policy_norm(policy_context))

        router_context = torch.cat([shared_hidden, views["summary"]], dim=-1)
        router_hidden = self.router_backbone(self.router_norm(router_context))
        router_logits = self.router_head(router_hidden) / max(float(self.router_temperature), 1e-6)
        router_probs = torch.softmax(router_logits, dim=-1)
        router_entropy = -(router_probs * torch.log(router_probs.clamp(min=1e-8))).sum(dim=-1)
        router_top2 = torch.topk(router_probs, k=min(2, self.num_experts), dim=-1, sorted=True).values
        router_top1 = router_top2[..., 0]
        router_second = router_top2[..., 1] if router_top2.shape[-1] > 1 else router_top1.new_zeros(router_top1.shape)
        router_margin = router_top1 - router_second

        expert_action_logits_list: list[torch.Tensor] = []
        expert_gain_pred_list: list[torch.Tensor] = []
        expert_hidden_list: list[torch.Tensor] = []
        expert_inputs = [
            self._build_expert_input(
                expert_index=0,
                shared_hidden=shared_hidden,
                summary=views["summary"],
                slot_pool=views["slot_pool"],
                token_pool=views["token_pool"],
                cluster_core=views["cluster_core"],
                assignment_summary=views["assignment_summary"],
                action_ctx=action_ctx.squeeze(0),
            ),
            self._build_expert_input(
                expert_index=1,
                shared_hidden=shared_hidden,
                summary=views["summary"],
                slot_pool=views["slot_pool"],
                token_pool=views["token_pool"],
                cluster_core=views["cluster_core"],
                assignment_summary=views["assignment_summary"],
                action_ctx=action_ctx.squeeze(0),
            ),
            self._build_expert_input(
                expert_index=2,
                shared_hidden=shared_hidden,
                summary=views["summary"],
                slot_pool=views["slot_pool"],
                token_pool=views["token_pool"],
                cluster_core=views["cluster_core"],
                assignment_summary=views["assignment_summary"],
                action_ctx=action_ctx.squeeze(0),
            ),
        ]
        for expert_index, (expert_norm, expert_backbone, action_head, gain_head, expert_input) in enumerate(
            zip(
                self.expert_norms,
                self.expert_backbones,
                self.expert_action_heads,
                self.expert_gain_heads,
                expert_inputs,
            )
        ):
            expert_hidden = expert_backbone(expert_norm(expert_input))
            expert_hidden_list.append(expert_hidden)
            expert_action_logits = action_head(expert_hidden) + self.expert_action_bias[expert_index]
            expert_gain_raw = gain_head(expert_hidden).squeeze(-1)
            expert_action_logits_list.append(expert_action_logits)
            expert_gain_pred_list.append(2.0 * torch.tanh(expert_gain_raw / 2.0))

        expert_action_logits_tensor = torch.stack(expert_action_logits_list, dim=0)
        expert_gain_pred_tensor = torch.stack(expert_gain_pred_list, dim=0)
        mixture_action_logits = torch.sum(router_probs.unsqueeze(-1) * expert_action_logits_tensor, dim=0)
        mixture_action_probs = torch.softmax(mixture_action_logits, dim=-1)
        mixture_gain_pred = torch.sum(router_probs * expert_gain_pred_tensor, dim=0)
        action_margin = mixture_action_probs[..., 0] - mixture_action_probs[..., 2]
        legacy_policy_score = mixture_gain_pred + 0.5 * action_margin + 0.25 * router_margin
        selection_features = self._build_selection_input(
            policy_context=policy_context,
            policy_hidden=shared_hidden,
            action_probs=mixture_action_probs,
            gain_pred=mixture_gain_pred,
            action_margin=action_margin,
            legacy_policy_score=legacy_policy_score,
            router_hidden=router_hidden,
            router_probs=router_probs,
            router_margin=router_margin,
            router_entropy=router_entropy,
        )
        selection_hidden = self.selection_backbone(self.selection_norm(selection_features))
        selection_raw = self.selection_head(selection_hidden).squeeze(-1)
        selection_score = 2.0 * torch.tanh(selection_raw / 2.0)
        calibration_raw = self.calibration_head(selection_hidden).squeeze(-1)
        calibration_score = 2.0 * torch.tanh(calibration_raw / 2.0)
        blend_gate_input = torch.cat(
            [
                selection_hidden.reshape(-1).to(dtype=torch.float32),
                router_hidden.reshape(-1).to(dtype=torch.float32),
                router_margin.reshape(-1).to(dtype=torch.float32),
                router_entropy.reshape(-1).to(dtype=torch.float32),
                legacy_policy_score.reshape(-1).to(dtype=torch.float32),
            ],
            dim=-1,
        )
        blend_gate_hidden = self.blend_gate_backbone(self.blend_gate_norm(blend_gate_input))
        blend_gate_raw = self.blend_gate_head(blend_gate_hidden).squeeze(-1)
        blend_gate = torch.sigmoid(blend_gate_raw)
        if self.policy_score_mode == "legacy_mixture":
            policy_score = legacy_policy_score
        elif self.policy_score_mode == "residual_selection":
            policy_score = legacy_policy_score + 0.5 * selection_score
        elif self.policy_score_mode == "calibrated_residual":
            policy_score = selection_score + 0.5 * calibration_score
        elif self.policy_score_mode == "gated_blend":
            policy_score = blend_gate * selection_score + (1.0 - blend_gate) * legacy_policy_score
        else:
            policy_score = selection_score

        outputs: Dict[str, torch.Tensor] = {
            "edge_logits": edge_logits,
            "defer_logits": defer_logits,
            "action_logits": mixture_action_logits,
            "action_probs": mixture_action_probs,
            "gain_raw": mixture_gain_pred,
            "gain_pred": mixture_gain_pred,
            "action_margin": action_margin,
            "selection_raw": selection_raw,
            "selection_score": selection_score,
            "calibration_raw": calibration_raw,
            "calibration_score": calibration_score,
            "blend_gate_raw": blend_gate_raw,
            "blend_gate": blend_gate,
            "legacy_policy_score": legacy_policy_score,
            "policy_score": policy_score,
            "decision_score": policy_score,
            "router_logits": router_logits,
            "router_probs": router_probs,
            "router_entropy": router_entropy,
            "router_margin": router_margin,
            "expert_action_logits": expert_action_logits_tensor,
            "expert_gain_pred": expert_gain_pred_tensor,
        }
        if return_context:
            outputs.update(
                {
                    "policy_features": policy_context,
                    "policy_hidden": shared_hidden,
                    "selection_features": selection_features,
                    "selection_hidden": selection_hidden,
                    "blend_gate_hidden": blend_gate_hidden,
                    "dense_logits": dense_logits,
                    "det_hidden": backbone_outputs["det_hidden"],
                    "track_hidden": backbone_outputs["track_hidden"],
                    "edge_hidden": backbone_outputs["edge_hidden"],
                    "cluster_hidden": backbone_outputs["cluster_hidden"],
                    "row_ctx": backbone_outputs["row_ctx"],
                    "col_ctx": backbone_outputs["col_ctx"],
                    "slot_tokens": slot_tokens.squeeze(0),
                    "token_tokens": token_tokens.squeeze(0),
                    "action_ctx": action_ctx.squeeze(0),
                    "router_hidden": router_hidden,
                    "expert_hidden": torch.stack(expert_hidden_list, dim=0),
                }
            )
        return outputs

    def checkpoint_payload(self, **extra: Any) -> Dict[str, Any]:
        payload = {
            "model_type": "action_policy",
            "model_arch": MODEL_ARCH_ROUTED_MOE,
            "model_state": self.state_dict(),
            "model_kwargs": {
                "det_dim": self.det_dim,
                "track_dim": self.track_dim,
                "edge_dim": self.edge_dim,
                "cluster_dim": self.cluster_dim,
                "hidden_dim": self.hidden_dim,
                "policy_hidden_dim": self.policy_hidden_dim,
                "dropout": self.dropout,
                "token_dim": self.token_dim,
                "num_slots": self.num_slots,
                "num_encoder_layers": self.num_encoder_layers,
                "num_heads": self.num_heads,
                "num_experts": self.num_experts,
                "router_hidden_dim": self.router_hidden_dim,
                "router_temperature": self.router_temperature,
                "policy_score_mode": self.policy_score_mode,
            },
            "feature_names": {
                "det": list(DET_FEATURE_NAMES),
                "track": list(TRACK_FEATURE_NAMES),
                "edge": list(EDGE_FEATURE_NAMES),
                "cluster": list(CLUSTER_FEATURE_NAMES),
            },
            "action_names": list(ACTION_NAMES),
        }
        payload.update(extra)
        return payload

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        *,
        map_location: str | torch.device | None = None,
    ) -> "GraphAssocCommitRoutedSetSlotPolicy":
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        model_kwargs = dict(checkpoint.get("model_kwargs", {}))
        model = cls(**model_kwargs)
        saved_policy_score_mode = str(
            checkpoint.get("policy_score_mode", model_kwargs.get("policy_score_mode", "learned_selection"))
            or model_kwargs.get("policy_score_mode", "learned_selection")
        ).strip().lower()
        if saved_policy_score_mode in {
            "learned_selection",
            "legacy_mixture",
            "residual_selection",
            "calibrated_residual",
            "gated_blend",
        }:
            model.policy_score_mode = saved_policy_score_mode
        state = checkpoint.get("model_state", checkpoint)
        model.load_state_dict(state, strict=False)
        model.eval()
        return model


class GraphAssocCommitHierarchicalRoutePolicy(GraphAssocCommitSetSlotPolicy):
    """
    Hierarchical commit policy:
    1. shared set/slot encoder builds a compact conflict view.
    2. a learned gate predicts whether the learned policy should override the base.
    3. a route prior and route-specific specialists refine rewrite / defer / reject.

    This keeps the original action labels intact while turning the old mixture
    into a structured residual router that is easier to interpret and train.
    """

    def __init__(
        self,
        det_dim: int = len(DET_FEATURE_NAMES),
        track_dim: int = len(TRACK_FEATURE_NAMES),
        edge_dim: int = len(EDGE_FEATURE_NAMES),
        cluster_dim: int = len(CLUSTER_FEATURE_NAMES),
        hidden_dim: int = 128,
        policy_hidden_dim: int | None = None,
        dropout: float = 0.1,
        token_dim: int | None = None,
        num_slots: int = 4,
        num_encoder_layers: int = 2,
        num_heads: int = 4,
        num_experts: int = 3,
        router_hidden_dim: int | None = None,
        gate_hidden_dim: int | None = None,
        route_temperature: float = 1.0,
        policy_score_mode: str = "gated_route_blend",
    ) -> None:
        if int(num_experts) != 3:
            raise ValueError("GraphAssocCommitHierarchicalRoutePolicy currently expects num_experts=3")
        super().__init__(
            det_dim=det_dim,
            track_dim=track_dim,
            edge_dim=edge_dim,
            cluster_dim=cluster_dim,
            hidden_dim=hidden_dim,
            policy_hidden_dim=policy_hidden_dim,
            dropout=dropout,
            token_dim=token_dim,
            num_slots=num_slots,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
        )
        self.num_experts = int(num_experts)
        self.router_hidden_dim = int(router_hidden_dim if router_hidden_dim is not None else self.policy_hidden_dim)
        self.gate_hidden_dim = int(gate_hidden_dim if gate_hidden_dim is not None else self.policy_hidden_dim)
        self.route_temperature = float(route_temperature if float(route_temperature) > 0.0 else 1.0)
        requested_policy_score_mode = str(policy_score_mode or "gated_route_blend").strip().lower()
        if requested_policy_score_mode in {"learned_selection", "gated_blend"}:
            requested_policy_score_mode = "gated_route_blend"
        if requested_policy_score_mode not in {
            "gated_route_blend",
            "route_residual",
            "base_policy",
            "final_action_margin",
        }:
            requested_policy_score_mode = "gated_route_blend"
        self.policy_score_mode = requested_policy_score_mode
        self.model_arch = MODEL_ARCH_HIER_ROUTE

        self.policy_context_dim = self.summary_dim + len(ACTION_NAMES) * self.token_dim
        route_input_dim = self.policy_context_dim + self.policy_hidden_dim + self.summary_dim
        self.route_norm = nn.LayerNorm(route_input_dim)
        self.route_backbone = nn.Sequential(
            nn.Linear(route_input_dim, self.router_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.router_hidden_dim, self.router_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.route_head = nn.Linear(self.router_hidden_dim, self.num_experts)

        gate_input_dim = self.policy_context_dim + self.policy_hidden_dim + self.num_experts + 2
        self.gate_norm = nn.LayerNorm(gate_input_dim)
        self.gate_backbone = nn.Sequential(
            nn.Linear(gate_input_dim, self.gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.gate_hidden_dim, self.gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.gate_head = nn.Linear(self.gate_hidden_dim, 1)

        expert_input_dims = [
            self.policy_hidden_dim + self.token_dim * 2 + 12,
            self.policy_hidden_dim + self.token_dim * 2 + self.hidden_dim + self.cluster_dim,
            self.policy_hidden_dim + self.token_dim * 3 + self.hidden_dim + self.cluster_dim + 12,
        ]
        self.expert_input_dims = list(expert_input_dims)
        self.expert_norms = nn.ModuleList([nn.LayerNorm(dim) for dim in expert_input_dims])
        self.expert_backbones = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, self.policy_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.policy_hidden_dim, self.policy_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                )
                for dim in expert_input_dims
            ]
        )
        self.expert_action_heads = nn.ModuleList([nn.Linear(self.policy_hidden_dim, len(ACTION_NAMES)) for _ in range(self.num_experts)])
        self.expert_gain_heads = nn.ModuleList([nn.Linear(self.policy_hidden_dim, 1) for _ in range(self.num_experts)])
        expert_bias = torch.tensor(
            [
                [0.30, -0.05, -0.25],
                [-0.05, 0.30, -0.05],
                [-0.25, -0.05, 0.30],
            ],
            dtype=torch.float32,
        )
        self.expert_action_bias = nn.Parameter(expert_bias[: self.num_experts].clone())

    @staticmethod
    def _extract_views(
        *,
        slot_tokens: torch.Tensor,
        token_tokens: torch.Tensor,
        cluster_hidden: torch.Tensor,
        cluster_features: torch.Tensor,
        edge_logits: torch.Tensor,
        defer_logits: torch.Tensor,
        dense_logits: torch.Tensor,
        num_tracks: int,
    ) -> Dict[str, torch.Tensor]:
        slot_pool = _pool_mean_max(slot_tokens)
        token_pool = _pool_mean_max(token_tokens) if token_tokens.numel() > 0 else token_tokens.new_zeros((slot_tokens.shape[-1] * 2,))
        assignment_summary = _assignment_summary(
            edge_logits=edge_logits,
            defer_logits=defer_logits,
            dense_logits=dense_logits,
            num_tracks=num_tracks,
        )
        cluster_core = torch.cat([cluster_hidden.view(-1).to(dtype=torch.float32), cluster_features.view(-1).to(dtype=torch.float32)], dim=-1)
        summary = torch.cat([slot_pool, token_pool, cluster_core, assignment_summary], dim=-1)
        return {
            "slot_pool": slot_pool,
            "token_pool": token_pool,
            "cluster_core": cluster_core,
            "assignment_summary": assignment_summary,
            "summary": summary,
        }

    def _build_expert_input(
        self,
        *,
        expert_index: int,
        shared_hidden: torch.Tensor,
        slot_pool: torch.Tensor,
        token_pool: torch.Tensor,
        cluster_core: torch.Tensor,
        assignment_summary: torch.Tensor,
        action_ctx: torch.Tensor,
    ) -> torch.Tensor:
        if int(expert_index) == 0:
            base = torch.cat([shared_hidden, slot_pool, assignment_summary], dim=-1)
        elif int(expert_index) == 1:
            base = torch.cat([shared_hidden, token_pool, cluster_core], dim=-1)
        else:
            base = torch.cat([shared_hidden, action_ctx.reshape(-1), cluster_core, assignment_summary], dim=-1)
        if base.numel() != int(self.expert_input_dims[int(expert_index)]):
            raise RuntimeError(
                f"expert input size mismatch for expert {int(expert_index)}: "
                f"expected {int(self.expert_input_dims[int(expert_index)])}, got {int(base.numel())}"
            )
        return base

    def forward(
        self,
        det_features: torch.Tensor,
        track_features: torch.Tensor,
        edge_features: torch.Tensor,
        edge_det_index: torch.Tensor,
        edge_track_index: torch.Tensor,
        cluster_features: torch.Tensor,
        *,
        return_context: bool = False,
    ) -> Dict[str, torch.Tensor]:
        base_outputs = super().forward(
            det_features=det_features,
            track_features=track_features,
            edge_features=edge_features,
            edge_det_index=edge_det_index,
            edge_track_index=edge_track_index,
            cluster_features=cluster_features,
            return_context=True,
        )

        edge_logits = base_outputs["edge_logits"]
        defer_logits = base_outputs["defer_logits"]
        dense_logits = LocalConflictCommitRefiner.build_dense_assignment_logits(
            num_detections=int(det_features.shape[0]),
            num_tracks=int(track_features.shape[0]),
            edge_logits=edge_logits,
            edge_det_index=edge_det_index,
            edge_track_index=edge_track_index,
            defer_logits=defer_logits,
        )
        slot_tokens = base_outputs["slot_tokens"]
        token_tokens = base_outputs["token_tokens"]
        views = self._extract_views(
            slot_tokens=slot_tokens,
            token_tokens=token_tokens,
            cluster_hidden=base_outputs["cluster_hidden"],
            cluster_features=cluster_features.to(dtype=torch.float32).view(-1),
            edge_logits=edge_logits,
            defer_logits=defer_logits,
            dense_logits=dense_logits,
            num_tracks=int(track_features.shape[0]),
        )

        policy_context = base_outputs["policy_features"]
        shared_hidden = base_outputs["policy_hidden"]
        action_ctx = base_outputs["action_ctx"]
        route_context = torch.cat([policy_context, shared_hidden, views["summary"]], dim=-1)
        route_hidden = self.route_backbone(self.route_norm(route_context))
        route_logits = self.route_head(route_hidden) / max(float(self.route_temperature), 1e-6)
        route_probs = torch.softmax(route_logits, dim=-1)
        route_entropy = -(route_probs * torch.log(route_probs.clamp(min=1e-8))).sum(dim=-1)
        route_top2 = torch.topk(route_probs, k=min(2, self.num_experts), dim=-1, sorted=True).values
        route_top1 = route_top2[..., 0]
        route_second = route_top2[..., 1] if route_top2.shape[-1] > 1 else route_top1.new_zeros(route_top1.shape)
        route_margin = route_top1 - route_second
        route_confidence = route_top1

        gate_input = torch.cat(
            [
                policy_context.reshape(-1).to(dtype=torch.float32),
                shared_hidden.reshape(-1).to(dtype=torch.float32),
                route_probs.reshape(-1).to(dtype=torch.float32),
                route_margin.reshape(-1).to(dtype=torch.float32),
                route_entropy.reshape(-1).to(dtype=torch.float32),
            ],
            dim=-1,
        )
        gate_hidden = self.gate_backbone(self.gate_norm(gate_input))
        gate_logit = self.gate_head(gate_hidden).squeeze(-1)
        gate_prob = torch.sigmoid(gate_logit)

        expert_action_logits_list: list[torch.Tensor] = []
        expert_gain_raw_list: list[torch.Tensor] = []
        expert_hidden_list: list[torch.Tensor] = []
        expert_inputs = [
            self._build_expert_input(
                expert_index=0,
                shared_hidden=shared_hidden,
                slot_pool=views["slot_pool"],
                token_pool=views["token_pool"],
                cluster_core=views["cluster_core"],
                assignment_summary=views["assignment_summary"],
                action_ctx=action_ctx,
            ),
            self._build_expert_input(
                expert_index=1,
                shared_hidden=shared_hidden,
                slot_pool=views["slot_pool"],
                token_pool=views["token_pool"],
                cluster_core=views["cluster_core"],
                assignment_summary=views["assignment_summary"],
                action_ctx=action_ctx,
            ),
            self._build_expert_input(
                expert_index=2,
                shared_hidden=shared_hidden,
                slot_pool=views["slot_pool"],
                token_pool=views["token_pool"],
                cluster_core=views["cluster_core"],
                assignment_summary=views["assignment_summary"],
                action_ctx=action_ctx,
            ),
        ]
        for expert_index, (expert_norm, expert_backbone, action_head, gain_head, expert_input) in enumerate(
            zip(
                self.expert_norms,
                self.expert_backbones,
                self.expert_action_heads,
                self.expert_gain_heads,
                expert_inputs,
            )
        ):
            expert_hidden = expert_backbone(expert_norm(expert_input))
            expert_hidden_list.append(expert_hidden)
            expert_action_logits = action_head(expert_hidden) + self.expert_action_bias[expert_index]
            expert_gain_raw = gain_head(expert_hidden).squeeze(-1)
            expert_action_logits_list.append(expert_action_logits)
            expert_gain_raw_list.append(expert_gain_raw)

        expert_action_logits_tensor = torch.stack(expert_action_logits_list, dim=0)
        expert_gain_raw_tensor = torch.stack(expert_gain_raw_list, dim=0)
        expert_gain_pred_tensor = 2.0 * torch.tanh(expert_gain_raw_tensor / 2.0)
        expert_action_mix_logits = torch.sum(route_probs.unsqueeze(-1) * expert_action_logits_tensor, dim=0)
        final_action_logits = (1.0 - gate_prob) * base_outputs["action_logits"] + gate_prob * expert_action_mix_logits
        final_action_probs = torch.softmax(final_action_logits, dim=-1)
        final_action_margin = final_action_probs[..., 0] - final_action_probs[..., 2]

        route_gain_raw = torch.sum(route_probs * expert_gain_raw_tensor, dim=0)
        route_gain_pred = torch.sum(route_probs * expert_gain_pred_tensor, dim=0)
        final_gain_raw = (1.0 - gate_prob) * base_outputs["gain_raw"] + gate_prob * route_gain_raw
        final_gain_pred = 2.0 * torch.tanh(final_gain_raw / 2.0)

        base_policy_score = base_outputs["policy_score"]
        route_score = final_action_margin + 0.5 * route_margin
        if self.policy_score_mode == "base_policy":
            policy_score = base_policy_score
        elif self.policy_score_mode == "route_residual":
            policy_score = base_policy_score + gate_prob * route_score
        elif self.policy_score_mode == "final_action_margin":
            policy_score = final_action_margin
        else:
            policy_score = gate_prob * route_score + (1.0 - gate_prob) * base_policy_score

        outputs: Dict[str, torch.Tensor] = {
            "edge_logits": edge_logits,
            "defer_logits": defer_logits,
            "base_action_logits": base_outputs["action_logits"],
            "base_action_probs": base_outputs["action_probs"],
            "base_gain_raw": base_outputs["gain_raw"],
            "base_gain_pred": base_outputs["gain_pred"],
            "route_logits": route_logits,
            "route_probs": route_probs,
            "router_logits": route_logits,
            "router_probs": route_probs,
            "router_entropy": route_entropy,
            "router_margin": route_margin,
            "route_confidence": route_confidence,
            "gate_logit": gate_logit,
            "gate_prob": gate_prob,
            "specialist_action_logits": expert_action_logits_tensor,
            "specialist_gain_raw": expert_gain_raw_tensor,
            "specialist_gain_pred": expert_gain_pred_tensor,
            "expert_action_logits": expert_action_logits_tensor,
            "expert_gain_pred": expert_gain_pred_tensor,
            "expert_action_mix_logits": expert_action_mix_logits,
            "action_logits": final_action_logits,
            "action_probs": final_action_probs,
            "gain_raw": final_gain_raw,
            "gain_pred": final_gain_pred,
            "action_margin": final_action_margin,
            "policy_score": policy_score,
            "decision_score": policy_score,
            "selection_score": policy_score,
            "legacy_policy_score": base_policy_score,
            "gate_gain_pred": route_gain_pred,
        }
        if return_context:
            outputs.update(
                {
                    "policy_features": policy_context,
                    "policy_hidden": shared_hidden,
                    "route_hidden": route_hidden,
                    "gate_hidden": gate_hidden,
                    "dense_logits": dense_logits,
                    "det_hidden": base_outputs["det_hidden"],
                    "track_hidden": base_outputs["track_hidden"],
                    "edge_hidden": base_outputs["edge_hidden"],
                    "cluster_hidden": base_outputs["cluster_hidden"],
                    "row_ctx": base_outputs["row_ctx"],
                    "col_ctx": base_outputs["col_ctx"],
                    "slot_tokens": slot_tokens,
                    "token_tokens": token_tokens,
                    "action_ctx": action_ctx,
                    "specialist_hidden": torch.stack(expert_hidden_list, dim=0),
                    "specialist_inputs": torch.stack([inp for inp in expert_inputs], dim=0),
                }
            )
        return outputs

    def checkpoint_payload(self, **extra: Any) -> Dict[str, Any]:
        payload = {
            "model_type": "action_policy",
            "model_arch": MODEL_ARCH_HIER_ROUTE,
            "model_state": self.state_dict(),
            "model_kwargs": {
                "det_dim": self.det_dim,
                "track_dim": self.track_dim,
                "edge_dim": self.edge_dim,
                "cluster_dim": self.cluster_dim,
                "hidden_dim": self.hidden_dim,
                "policy_hidden_dim": self.policy_hidden_dim,
                "dropout": self.dropout,
                "token_dim": self.token_dim,
                "num_slots": self.num_slots,
                "num_encoder_layers": self.num_encoder_layers,
                "num_heads": self.num_heads,
                "num_experts": self.num_experts,
                "router_hidden_dim": self.router_hidden_dim,
                "gate_hidden_dim": self.gate_hidden_dim,
                "route_temperature": self.route_temperature,
                "policy_score_mode": self.policy_score_mode,
            },
            "feature_names": {
                "det": list(DET_FEATURE_NAMES),
                "track": list(TRACK_FEATURE_NAMES),
                "edge": list(EDGE_FEATURE_NAMES),
                "cluster": list(CLUSTER_FEATURE_NAMES),
            },
            "action_names": list(ACTION_NAMES),
        }
        payload.update(extra)
        return payload

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        *,
        map_location: str | torch.device | None = None,
    ) -> "GraphAssocCommitHierarchicalRoutePolicy":
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        model_kwargs = dict(checkpoint.get("model_kwargs", {}))
        model = cls(**model_kwargs)
        saved_policy_score_mode = str(
            checkpoint.get("policy_score_mode", model_kwargs.get("policy_score_mode", "gated_route_blend"))
            or model_kwargs.get("policy_score_mode", "gated_route_blend")
        ).strip().lower()
        if saved_policy_score_mode in {
            "gated_route_blend",
            "route_residual",
            "base_policy",
            "final_action_margin",
            "learned_selection",
            "gated_blend",
        }:
            if saved_policy_score_mode in {"learned_selection", "gated_blend"}:
                model.policy_score_mode = "gated_route_blend"
            else:
                model.policy_score_mode = saved_policy_score_mode
        state = checkpoint.get("model_state", checkpoint)
        model.load_state_dict(state, strict=False)
        model.eval()
        return model
