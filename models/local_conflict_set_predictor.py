from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import torch
from torch import nn


MODEL_FAMILY = "set_predictor_v2"
FEATURE_VERSION = "v2_hostnorm_geom"
DECODER_FAMILY = "hungarian_private_defer"

DET_FEATURE_NAMES = (
    "det_score_raw",
    "row_degree",
    "row_top1_minus_top2",
    "row_entropy",
    "det_cx",
    "det_cy",
    "det_log_w",
    "det_log_h",
    "det_aspect",
)

TRACK_FEATURE_NAMES = (
    "track_gap_log1p",
    "track_hist_len_log1p",
    "col_degree",
    "track_cx",
    "track_cy",
    "track_log_w",
    "track_log_h",
    "track_aspect",
)

EDGE_FEATURE_NAMES = (
    "base_score_raw",
    "refined_score_raw",
    "motion_score_raw",
    "base_score_row_z",
    "refined_score_row_z",
    "motion_score_row_z",
    "refined_score_row_softmax",
    "refined_gap_to_row_top1",
    "rank_frac",
    "refined_score_col_z",
    "refined_minus_base",
    "motion_minus_refined",
    "iou",
    "bbox_dist_score",
    "delta_cx_norm",
    "delta_cy_norm",
    "delta_log_w",
    "delta_log_h",
)

CLUSTER_FEATURE_NAMES = (
    "num_dets",
    "num_tracks",
    "num_edges",
    "mean_row_degree",
    "max_row_degree",
    "mean_col_degree",
    "max_col_degree",
    "mean_row_entropy",
    "max_row_entropy",
    "mean_refined_gap",
    "max_refined_gap",
)


def normalize_host_vocab(host_vocab: Sequence[str] | None) -> list[str]:
    vocab = [str(token).strip() for token in list(host_vocab or []) if str(token).strip()]
    if not vocab:
        return ["unknown"]
    deduped: list[str] = []
    seen: set[str] = set()
    for token in vocab:
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return deduped or ["unknown"]


def encode_host_variant(host_variant: str, host_vocab: Sequence[str] | None) -> int:
    vocab = normalize_host_vocab(host_vocab)
    token = str(host_variant or "").strip()
    if token in vocab:
        return int(vocab.index(token))
    return 0


def zscore_1d(values: torch.Tensor) -> torch.Tensor:
    values = values.to(dtype=torch.float32)
    if values.numel() <= 1:
        return torch.zeros_like(values)
    mean = values.mean()
    std = values.std(unbiased=False)
    if float(std.item()) < 1e-6:
        return torch.zeros_like(values)
    return (values - mean) / std


def softmax_probs_1d(values: torch.Tensor) -> torch.Tensor:
    values = values.to(dtype=torch.float32)
    if values.numel() == 0:
        return values
    return torch.softmax(values - values.max(), dim=0)


def entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    probs = probs.to(dtype=torch.float32)
    if probs.numel() == 0:
        return probs.new_zeros(())
    return -(probs * torch.log(probs.clamp(min=1e-8))).sum()


def pair_geometry_features(
    det_box_cxcywh: torch.Tensor,
    track_box_cxcywh: torch.Tensor,
    *,
    tau: float = 1.0,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    det = det_box_cxcywh.to(dtype=torch.float32)
    track = track_box_cxcywh.to(dtype=torch.float32)
    det_cx, det_cy, det_w, det_h = det.unbind(dim=-1)
    track_cx, track_cy, track_w, track_h = track.unbind(dim=-1)
    det_w = det_w.clamp(min=eps)
    det_h = det_h.clamp(min=eps)
    track_w = track_w.clamp(min=eps)
    track_h = track_h.clamp(min=eps)

    delta_cx = (det_cx - track_cx) / track_w
    delta_cy = (det_cy - track_cy) / track_h
    delta_log_w = torch.log(det_w / track_w)
    delta_log_h = torch.log(det_h / track_h)
    dist = torch.sqrt(delta_cx.square() + delta_cy.square() + delta_log_w.square() + delta_log_h.square() + eps)
    bbox_dist_score = torch.exp(-dist / max(float(tau), eps))

    det_x1 = det_cx - 0.5 * det_w
    det_y1 = det_cy - 0.5 * det_h
    det_x2 = det_cx + 0.5 * det_w
    det_y2 = det_cy + 0.5 * det_h
    track_x1 = track_cx - 0.5 * track_w
    track_y1 = track_cy - 0.5 * track_h
    track_x2 = track_cx + 0.5 * track_w
    track_y2 = track_cy + 0.5 * track_h
    inter_w = (torch.minimum(det_x2, track_x2) - torch.maximum(det_x1, track_x1)).clamp(min=0.0)
    inter_h = (torch.minimum(det_y2, track_y2) - torch.maximum(det_y1, track_y1)).clamp(min=0.0)
    inter = inter_w * inter_h
    det_area = (det_x2 - det_x1).clamp(min=0.0) * (det_y2 - det_y1).clamp(min=0.0)
    track_area = (track_x2 - track_x1).clamp(min=0.0) * (track_y2 - track_y1).clamp(min=0.0)
    union = det_area + track_area - inter + eps
    iou = inter / union

    return {
        "iou": iou,
        "bbox_dist_score": bbox_dist_score,
        "delta_cx_norm": delta_cx,
        "delta_cy_norm": delta_cy,
        "delta_log_w": delta_log_w,
        "delta_log_h": delta_log_h,
    }


def _mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    dropout: float,
    *,
    final_activation: bool = False,
) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Linear(int(input_dim), int(hidden_dim)),
        nn.GELU(),
        nn.Dropout(float(dropout)),
        nn.Linear(int(hidden_dim), int(output_dim)),
    ]
    if final_activation:
        layers.extend([nn.GELU(), nn.Dropout(float(dropout))])
    return nn.Sequential(*layers)


def _segment_mean_max(
    values: torch.Tensor,
    index: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    if num_segments <= 0:
        return values.new_zeros((0, values.shape[-1] * 2))
    feature_dim = int(values.shape[-1])
    if values.numel() == 0:
        return values.new_zeros((num_segments, feature_dim * 2))
    pooled = []
    for seg_idx in range(int(num_segments)):
        mask = index == int(seg_idx)
        if bool(mask.any().item()):
            seg_vals = values[mask]
            pooled.append(torch.cat([seg_vals.mean(dim=0), seg_vals.max(dim=0).values], dim=-1))
        else:
            pooled.append(values.new_zeros((feature_dim * 2,)))
    return torch.stack(pooled, dim=0)


@dataclass
class LocalConflictSetPredictorConfig:
    det_dim: int = len(DET_FEATURE_NAMES)
    track_dim: int = len(TRACK_FEATURE_NAMES)
    edge_dim: int = len(EDGE_FEATURE_NAMES)
    cluster_dim: int = len(CLUSTER_FEATURE_NAMES)
    hidden_dim: int = 128
    num_heads: int = 4
    num_conflict_blocks: int = 2
    dropout: float = 0.1
    num_host_variants: int = 1
    raw_edge_score_dims: int = 3


class _ConflictBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.row_attn = nn.MultiheadAttention(self.hidden_dim, int(num_heads), dropout=float(dropout), batch_first=True)
        self.col_attn = nn.MultiheadAttention(self.hidden_dim, int(num_heads), dropout=float(dropout), batch_first=True)
        self.row_norm = nn.LayerNorm(self.hidden_dim)
        self.col_norm = nn.LayerNorm(self.hidden_dim)
        self.det_norm = nn.LayerNorm(self.hidden_dim)
        self.track_norm = nn.LayerNorm(self.hidden_dim)
        self.edge_norm = nn.LayerNorm(self.hidden_dim)
        self.det_update = _mlp(self.hidden_dim * 4, self.hidden_dim, self.hidden_dim, dropout)
        self.track_update = _mlp(self.hidden_dim * 4, self.hidden_dim, self.hidden_dim, dropout)
        self.edge_update = _mlp(self.hidden_dim * 8, self.hidden_dim, self.hidden_dim, dropout)

    def _segment_attention(
        self,
        tokens: torch.Tensor,
        segment_index: torch.Tensor,
        num_segments: int,
        attn: nn.MultiheadAttention,
        norm: nn.LayerNorm,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.numel() == 0:
            return tokens, tokens.new_zeros((int(num_segments), self.hidden_dim * 2))
        updated = tokens.clone()
        for seg_idx in range(int(num_segments)):
            member_index = (segment_index == int(seg_idx)).nonzero(as_tuple=True)[0]
            if member_index.numel() == 0:
                continue
            seq = tokens.index_select(0, member_index).unsqueeze(0)
            attn_out, _ = attn(seq, seq, seq, need_weights=False)
            seg_updated = norm(seq.squeeze(0) + attn_out.squeeze(0))
            updated.index_copy_(0, member_index, seg_updated)
        pooled = _segment_mean_max(updated, segment_index, num_segments)
        return updated, pooled

    def forward(
        self,
        det_tokens: torch.Tensor,
        track_tokens: torch.Tensor,
        edge_tokens: torch.Tensor,
        edge_det_index: torch.Tensor,
        edge_track_index: torch.Tensor,
        cluster_token: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_dets = int(det_tokens.shape[0])
        num_tracks = int(track_tokens.shape[0])
        cluster_for_dets = cluster_token.view(1, -1).expand(num_dets, -1)
        cluster_for_tracks = cluster_token.view(1, -1).expand(num_tracks, -1)

        if edge_tokens.numel() == 0:
            empty_row = det_tokens.new_zeros((num_dets, self.hidden_dim * 2))
            empty_col = track_tokens.new_zeros((num_tracks, self.hidden_dim * 2))
            det_tokens = self.det_norm(
                det_tokens + self.det_update(torch.cat([det_tokens, empty_row, cluster_for_dets], dim=-1))
            )
            track_tokens = self.track_norm(
                track_tokens + self.track_update(torch.cat([track_tokens, empty_col, cluster_for_tracks], dim=-1))
            )
            return det_tokens, track_tokens, edge_tokens, empty_row, empty_col

        row_tokens, row_pool = self._segment_attention(
            edge_tokens,
            edge_det_index,
            num_dets,
            self.row_attn,
            self.row_norm,
        )
        col_tokens, col_pool = self._segment_attention(
            row_tokens,
            edge_track_index,
            num_tracks,
            self.col_attn,
            self.col_norm,
        )

        det_tokens = self.det_norm(
            det_tokens + self.det_update(torch.cat([det_tokens, row_pool, cluster_for_dets], dim=-1))
        )
        track_tokens = self.track_norm(
            track_tokens + self.track_update(torch.cat([track_tokens, col_pool, cluster_for_tracks], dim=-1))
        )

        cluster_for_edges = cluster_token.view(1, -1).expand(int(edge_tokens.shape[0]), -1)
        edge_inputs = torch.cat(
            [
                col_tokens,
                det_tokens.index_select(0, edge_det_index),
                track_tokens.index_select(0, edge_track_index),
                row_pool.index_select(0, edge_det_index),
                col_pool.index_select(0, edge_track_index),
                cluster_for_edges,
            ],
            dim=-1,
        )
        edge_tokens = self.edge_norm(col_tokens + self.edge_update(edge_inputs))
        return det_tokens, track_tokens, edge_tokens, row_pool, col_pool


class HostConditionedLocalConflictSetPredictor(nn.Module):
    def __init__(
        self,
        det_dim: int = len(DET_FEATURE_NAMES),
        track_dim: int = len(TRACK_FEATURE_NAMES),
        edge_dim: int = len(EDGE_FEATURE_NAMES),
        cluster_dim: int = len(CLUSTER_FEATURE_NAMES),
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_conflict_blocks: int = 2,
        dropout: float = 0.1,
        num_host_variants: int = 1,
        raw_edge_score_dims: int = 3,
    ) -> None:
        super().__init__()
        self.det_dim = int(det_dim)
        self.track_dim = int(track_dim)
        self.edge_dim = int(edge_dim)
        self.cluster_dim = int(cluster_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.num_conflict_blocks = int(num_conflict_blocks)
        self.dropout = float(dropout)
        self.num_host_variants = max(int(num_host_variants), 1)
        self.raw_edge_score_dims = max(int(raw_edge_score_dims), 0)
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim={self.hidden_dim} must be divisible by num_heads={self.num_heads} for attention blocks"
            )

        self.det_encoder = _mlp(self.det_dim, self.hidden_dim, self.hidden_dim, self.dropout, final_activation=True)
        self.track_encoder = _mlp(
            self.track_dim,
            self.hidden_dim,
            self.hidden_dim,
            self.dropout,
            final_activation=True,
        )
        self.cluster_encoder = _mlp(
            self.cluster_dim,
            self.hidden_dim,
            self.hidden_dim,
            self.dropout,
            final_activation=True,
        )
        self.host_embedding = nn.Embedding(self.num_host_variants, self.hidden_dim)
        self.host_score_affine = nn.Linear(self.hidden_dim, self.raw_edge_score_dims * 2)
        self.edge_encoder = _mlp(
            self.edge_dim + self.hidden_dim * 2,
            self.hidden_dim,
            self.hidden_dim,
            self.dropout,
            final_activation=True,
        )
        self.blocks = nn.ModuleList(
            [_ConflictBlock(self.hidden_dim, self.num_heads, self.dropout) for _ in range(self.num_conflict_blocks)]
        )
        self.edge_head = _mlp(self.hidden_dim * 8, self.hidden_dim, 1, self.dropout)
        self.defer_head = _mlp(self.hidden_dim * 4, self.hidden_dim, 1, self.dropout)
        self.cluster_gate_head = _mlp(self.hidden_dim * 3, self.hidden_dim, 1, self.dropout)

        self.host_vocab = ["unknown"]
        self.feature_stats: Dict[str, Any] = {}
        self._reset_stable_init()

    def _reset_stable_init(self) -> None:
        # Host conditioning should start as an identity-like perturbation, not a random
        # remapping of raw host scores. This makes the first few epochs much less brittle.
        nn.init.zeros_(self.host_embedding.weight)
        nn.init.zeros_(self.host_score_affine.weight)
        nn.init.zeros_(self.host_score_affine.bias)

    def _coerce_host_index(
        self,
        host_variant_id: int | torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        if host_variant_id is None:
            return torch.zeros((1,), device=device, dtype=torch.long)
        if torch.is_tensor(host_variant_id):
            host_index = host_variant_id.to(device=device, dtype=torch.long).view(-1)
            if host_index.numel() == 0:
                return torch.zeros((1,), device=device, dtype=torch.long)
            return host_index[:1]
        return torch.tensor([int(host_variant_id)], device=device, dtype=torch.long)

    def _apply_host_conditioning(
        self,
        edge_features: torch.Tensor,
        host_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if edge_features.numel() == 0 or self.raw_edge_score_dims <= 0:
            return edge_features
        affine = self.host_score_affine(host_embedding.view(1, -1)).view(-1)
        gamma = 0.10 * affine[: self.raw_edge_score_dims].tanh()
        beta = 0.10 * affine[self.raw_edge_score_dims :].tanh()
        raw_scores = edge_features[:, : self.raw_edge_score_dims]
        conditioned_raw = raw_scores * (1.0 + gamma.view(1, -1)) + beta.view(1, -1)
        conditioned_raw = conditioned_raw.clamp(min=-8.0, max=8.0)
        if edge_features.shape[1] <= self.raw_edge_score_dims:
            return conditioned_raw
        return torch.cat([conditioned_raw, edge_features[:, self.raw_edge_score_dims :]], dim=-1)

    def forward(
        self,
        det_features: torch.Tensor,
        track_features: torch.Tensor,
        edge_features: torch.Tensor,
        edge_det_index: torch.Tensor,
        edge_track_index: torch.Tensor,
        cluster_features: torch.Tensor,
        host_variant_id: int | torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        det_features = det_features.to(dtype=torch.float32)
        track_features = track_features.to(dtype=torch.float32)
        edge_features = edge_features.to(dtype=torch.float32)
        edge_det_index = edge_det_index.to(dtype=torch.long)
        edge_track_index = edge_track_index.to(dtype=torch.long)
        cluster_features = cluster_features.to(dtype=torch.float32).view(1, -1)

        device = det_features.device
        host_index = self._coerce_host_index(host_variant_id, device)
        host_index = host_index.clamp(min=0, max=max(self.num_host_variants - 1, 0))
        host_hidden = self.host_embedding(host_index).view(-1)

        det_tokens = self.det_encoder(det_features)
        track_tokens = self.track_encoder(track_features)
        cluster_token = self.cluster_encoder(cluster_features).squeeze(0) + host_hidden

        edge_conditioned = self._apply_host_conditioning(edge_features, host_hidden)
        if edge_conditioned.shape[0] > 0:
            edge_inputs = torch.cat(
                [
                    edge_conditioned,
                    det_tokens.index_select(0, edge_det_index),
                    track_tokens.index_select(0, edge_track_index),
                ],
                dim=-1,
            )
            edge_tokens = self.edge_encoder(edge_inputs)
        else:
            edge_tokens = det_tokens.new_zeros((0, self.hidden_dim))

        row_pool = det_tokens.new_zeros((int(det_tokens.shape[0]), self.hidden_dim * 2))
        col_pool = track_tokens.new_zeros((int(track_tokens.shape[0]), self.hidden_dim * 2))
        for block in self.blocks:
            det_tokens, track_tokens, edge_tokens, row_pool, col_pool = block(
                det_tokens=det_tokens,
                track_tokens=track_tokens,
                edge_tokens=edge_tokens,
                edge_det_index=edge_det_index,
                edge_track_index=edge_track_index,
                cluster_token=cluster_token,
            )

        if edge_tokens.shape[0] > 0:
            cluster_for_edges = cluster_token.view(1, -1).expand(int(edge_tokens.shape[0]), -1)
            edge_logits = self.edge_head(
                torch.cat(
                    [
                        edge_tokens,
                        det_tokens.index_select(0, edge_det_index),
                        track_tokens.index_select(0, edge_track_index),
                        row_pool.index_select(0, edge_det_index),
                        col_pool.index_select(0, edge_track_index),
                        cluster_for_edges,
                    ],
                    dim=-1,
                )
            ).squeeze(-1)
        else:
            edge_logits = det_tokens.new_zeros((0,))
        edge_logits = edge_logits.clamp(min=-30.0, max=30.0)

        cluster_for_dets = cluster_token.view(1, -1).expand(int(det_tokens.shape[0]), -1)
        defer_logits = self.defer_head(
            torch.cat(
                [
                    det_tokens,
                    row_pool,
                    cluster_for_dets,
                ],
                dim=-1,
            )
        ).squeeze(-1)
        defer_logits = defer_logits.clamp(min=-30.0, max=30.0)

        det_summary = det_tokens.mean(dim=0) if det_tokens.shape[0] > 0 else cluster_token.new_zeros((self.hidden_dim,))
        edge_summary = edge_tokens.mean(dim=0) if edge_tokens.shape[0] > 0 else cluster_token.new_zeros((self.hidden_dim,))
        cluster_commit_logit = self.cluster_gate_head(
            torch.cat([cluster_token, det_summary, edge_summary], dim=-1).view(1, -1)
        ).view(())
        cluster_commit_logit = cluster_commit_logit.clamp(min=-30.0, max=30.0)

        return {
            "edge_logits": edge_logits,
            "defer_logits": defer_logits,
            "cluster_commit_logit": cluster_commit_logit,
            "cluster_utility_logit": cluster_commit_logit,
        }

    @staticmethod
    def build_dense_assignment_logits(
        *,
        num_detections: int,
        num_tracks: int,
        edge_logits: torch.Tensor,
        edge_det_index: torch.Tensor,
        edge_track_index: torch.Tensor,
        defer_logits: torch.Tensor,
        fill_value: float = -1e6,
    ) -> torch.Tensor:
        dense = defer_logits.new_full((int(num_detections), int(num_tracks) + 1), float(fill_value))
        if edge_logits.numel() > 0:
            dense[edge_det_index, edge_track_index] = edge_logits
        dense[:, int(num_tracks)] = defer_logits
        return dense

    def checkpoint_payload(self, **extra: Any) -> Dict[str, Any]:
        payload = {
            "model_family": MODEL_FAMILY,
            "feature_version": FEATURE_VERSION,
            "decoder": DECODER_FAMILY,
            "model_state": self.state_dict(),
            "model_kwargs": {
                "det_dim": self.det_dim,
                "track_dim": self.track_dim,
                "edge_dim": self.edge_dim,
                "cluster_dim": self.cluster_dim,
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "num_conflict_blocks": self.num_conflict_blocks,
                "dropout": self.dropout,
                "num_host_variants": self.num_host_variants,
                "raw_edge_score_dims": self.raw_edge_score_dims,
            },
            "feature_names": {
                "det": list(DET_FEATURE_NAMES),
                "track": list(TRACK_FEATURE_NAMES),
                "edge": list(EDGE_FEATURE_NAMES),
                "cluster": list(CLUSTER_FEATURE_NAMES),
            },
            "host_vocab": list(getattr(self, "host_vocab", ["unknown"])),
            "feature_stats": dict(getattr(self, "feature_stats", {})),
        }
        payload.update(extra)
        return payload

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        *,
        map_location: str | torch.device | None = None,
    ) -> "HostConditionedLocalConflictSetPredictor":
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        model_kwargs = dict(checkpoint.get("model_kwargs", {}))
        model = cls(**model_kwargs)
        state = checkpoint.get("model_state", checkpoint)
        model.load_state_dict(state, strict=True)
        model.host_vocab = normalize_host_vocab(checkpoint.get("host_vocab", ["unknown"]))
        model.feature_stats = dict(checkpoint.get("feature_stats", {}))
        model.eval()
        return model
