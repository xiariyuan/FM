from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass
from typing import Any, Dict

import torch
from torch import nn


DET_FEATURE_NAMES = (
    "det_score",
    "row_degree",
    "row_top1_minus_top2",
    "row_entropy",
)

TRACK_FEATURE_NAMES = (
    "track_gap",
    "track_hist_len",
    "col_degree",
)

EDGE_FEATURE_NAMES = (
    "base_score",
    "refined_score",
    "motion_score",
    "track_gap",
    "track_hist_len",
    "track_rank_frac",
)

CLUSTER_FEATURE_NAMES = (
    "num_dets",
    "num_tracks",
    "num_edges",
    "mean_row_degree",
    "max_row_degree",
    "mean_col_degree",
    "max_col_degree",
    "utility_gain",
    "cost_delta",
    "baseline_cost",
    "chosen_cost",
    "introduced_row_count",
    "suppressed_row_count",
    "recent_owner_row_count",
    "protected_young_active_row_count",
)


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
        nn.ReLU(inplace=True),
    )


def _segment_mean_max(
    values: torch.Tensor,
    index: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    if num_segments <= 0:
        return values.new_zeros((0, values.shape[-1] * 2))
    feat_dim = int(values.shape[-1])
    if values.numel() == 0:
        mean_out = values.new_zeros((num_segments, feat_dim))
        max_out = values.new_zeros((num_segments, feat_dim))
        return torch.cat([mean_out, max_out], dim=-1)

    segments = []
    for seg_idx in range(int(num_segments)):
        mask = index == int(seg_idx)
        if bool(mask.any().item()):
            seg_vals = values[mask]
            seg_mean = seg_vals.mean(dim=0)
            seg_max = seg_vals.max(dim=0).values
        else:
            seg_mean = values.new_zeros((feat_dim,))
            seg_max = values.new_zeros((feat_dim,))
        segments.append(torch.cat([seg_mean, seg_max], dim=-1))
    return torch.stack(segments, dim=0)


@dataclass
class LocalConflictCommitConfig:
    det_dim: int = len(DET_FEATURE_NAMES)
    track_dim: int = len(TRACK_FEATURE_NAMES)
    edge_dim: int = len(EDGE_FEATURE_NAMES)
    cluster_dim: int = len(CLUSTER_FEATURE_NAMES)
    hidden_dim: int = 128
    dropout: float = 0.1


class LocalConflictCommitRefiner(nn.Module):
    def __init__(
        self,
        det_dim: int = len(DET_FEATURE_NAMES),
        track_dim: int = len(TRACK_FEATURE_NAMES),
        edge_dim: int = len(EDGE_FEATURE_NAMES),
        cluster_dim: int = len(CLUSTER_FEATURE_NAMES),
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.det_dim = int(det_dim)
        self.track_dim = int(track_dim)
        self.edge_dim = int(edge_dim)
        self.cluster_dim = int(cluster_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)

        self.det_mlp = _mlp(self.det_dim, self.hidden_dim, self.hidden_dim, self.dropout)
        self.track_mlp = _mlp(self.track_dim, self.hidden_dim, self.hidden_dim, self.dropout)
        self.cluster_mlp = _mlp(self.cluster_dim, self.hidden_dim, self.hidden_dim, self.dropout)
        self.edge_mlp = _mlp(self.edge_dim + 2 * self.hidden_dim, self.hidden_dim, self.hidden_dim, self.dropout)

        pooled_dim = self.hidden_dim * 2
        self.edge_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 4 + pooled_dim * 2, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        self.defer_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + pooled_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
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
        det_features = det_features.to(dtype=torch.float32)
        track_features = track_features.to(dtype=torch.float32)
        edge_features = edge_features.to(dtype=torch.float32)
        edge_det_index = edge_det_index.to(dtype=torch.long)
        edge_track_index = edge_track_index.to(dtype=torch.long)
        cluster_features = cluster_features.to(dtype=torch.float32).view(1, -1)

        det_hidden = self.det_mlp(det_features)
        track_hidden = self.track_mlp(track_features)
        cluster_hidden = self.cluster_mlp(cluster_features).squeeze(0)

        if edge_features.shape[0] > 0:
            edge_input = torch.cat(
                [
                    edge_features,
                    det_hidden.index_select(0, edge_det_index),
                    track_hidden.index_select(0, edge_track_index),
                ],
                dim=-1,
            )
            edge_hidden = self.edge_mlp(edge_input)
        else:
            edge_hidden = det_hidden.new_zeros((0, self.hidden_dim))

        row_ctx = _segment_mean_max(edge_hidden, edge_det_index, det_hidden.shape[0])
        col_ctx = _segment_mean_max(edge_hidden, edge_track_index, track_hidden.shape[0])
        cluster_for_edges = cluster_hidden.view(1, -1).expand(edge_hidden.shape[0], -1)
        cluster_for_dets = cluster_hidden.view(1, -1).expand(det_hidden.shape[0], -1)

        if edge_hidden.shape[0] > 0:
            edge_logits = self.edge_head(
                torch.cat(
                    [
                        edge_hidden,
                        det_hidden.index_select(0, edge_det_index),
                        track_hidden.index_select(0, edge_track_index),
                        row_ctx.index_select(0, edge_det_index),
                        col_ctx.index_select(0, edge_track_index),
                        cluster_for_edges,
                    ],
                    dim=-1,
                )
            ).squeeze(-1)
        else:
            edge_logits = det_hidden.new_zeros((0,))

        defer_logits = self.defer_head(
            torch.cat(
                [
                    det_hidden,
                    row_ctx,
                    cluster_for_dets,
                ],
                dim=-1,
            )
        ).squeeze(-1)

        return {
            "edge_logits": edge_logits,
            "defer_logits": defer_logits,
            **(
                {
                    "det_hidden": det_hidden,
                    "track_hidden": track_hidden,
                    "edge_hidden": edge_hidden,
                    "cluster_hidden": cluster_hidden,
                    "row_ctx": row_ctx,
                    "col_ctx": col_ctx,
                }
                if return_context
                else {}
            ),
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
            "model_state": self.state_dict(),
            "model_kwargs": {
                "det_dim": self.det_dim,
                "track_dim": self.track_dim,
                "edge_dim": self.edge_dim,
                "cluster_dim": self.cluster_dim,
                "hidden_dim": self.hidden_dim,
                "dropout": self.dropout,
            },
            "feature_names": {
                "det": list(DET_FEATURE_NAMES),
                "track": list(TRACK_FEATURE_NAMES),
                "edge": list(EDGE_FEATURE_NAMES),
                "cluster": list(CLUSTER_FEATURE_NAMES),
            },
        }
        payload.update(extra)
        return payload

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        *,
        map_location: str | torch.device | None = None,
    ) -> "LocalConflictCommitRefiner":
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        model_kwargs = dict(checkpoint.get("model_kwargs", {}))
        signature = inspect.signature(cls.__init__)
        valid_kwargs: Dict[str, Any] = {}
        ignored_kwargs: Dict[str, Any] = {}
        for key, value in model_kwargs.items():
            if key in signature.parameters and key != "self":
                valid_kwargs[key] = value
            else:
                ignored_kwargs[key] = value
        if ignored_kwargs:
            warnings.warn(
                "Ignoring legacy checkpoint kwargs for LocalConflictCommitRefiner: "
                + ", ".join(sorted(ignored_kwargs)),
                RuntimeWarning,
                stacklevel=2,
            )
        model = cls(**valid_kwargs)
        state = checkpoint.get("model_state", checkpoint)
        model.load_state_dict(state, strict=True)
        model.eval()
        return model
