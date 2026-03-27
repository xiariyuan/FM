from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Sequence

import torch
from torch import nn


MODEL_FAMILY = "posthost_one_edit_scorer_v1"

CANDIDATE_FEATURE_NAMES = (
    "action_keep",
    "action_add",
    "action_swap",
    "action_defer",
    "num_dets",
    "num_tracks",
    "num_edges",
    "is_large_component",
    "host_pair_count",
    "host_pos_pair_count",
    "host_neg_pair_count",
    "host_score",
    "add_refined_score",
    "add_base_score",
    "add_iou",
    "add_bbox_dist",
    "add_row_degree",
    "add_row_margin",
    "add_row_entropy",
    "add_col_degree",
    "add_track_gap",
    "add_track_hist",
    "add_delta_cx",
    "add_delta_cy",
    "add_delta_log_w",
    "add_delta_log_h",
    "remove_refined_score",
    "remove_base_score",
    "remove_iou",
    "remove_bbox_dist",
    "remove_row_degree",
    "remove_row_margin",
    "remove_row_entropy",
    "remove_col_degree",
    "remove_track_gap",
    "remove_track_hist",
    "remove_delta_cx",
    "remove_delta_cy",
    "remove_delta_log_w",
    "remove_delta_log_h",
    "delta_refined_score",
    "delta_base_score",
    "delta_iou",
    "delta_bbox_dist",
    "delta_row_margin",
    "delta_row_entropy",
    "delta_track_gap",
    "delta_track_hist",
    "candidate_remove_count",
)


@dataclass
class PosthostOneEditScorerConfig:
    input_dim: int = len(CANDIDATE_FEATURE_NAMES)
    hidden_dim: int = 128
    dropout: float = 0.1
    num_layers: int = 3


class PosthostOneEditScorer(nn.Module):
    def __init__(
        self,
        input_dim: int = len(CANDIDATE_FEATURE_NAMES),
        hidden_dim: int = 128,
        dropout: float = 0.1,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.config = PosthostOneEditScorerConfig(
            input_dim=int(input_dim),
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
            num_layers=int(num_layers),
        )
        layers: list[nn.Module] = [nn.LayerNorm(self.config.input_dim)]
        in_dim = self.config.input_dim
        for _ in range(max(self.config.num_layers - 1, 1)):
            layers.extend(
                [
                    nn.Linear(in_dim, self.config.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(self.config.dropout),
                ]
            )
            in_dim = self.config.hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.scorer = nn.Sequential(*layers)

    def forward(self, candidate_features: torch.Tensor) -> torch.Tensor:
        feats = candidate_features.to(dtype=torch.float32)
        logits = self.scorer(feats).squeeze(-1)
        return logits.clamp(min=-30.0, max=30.0)

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        extra: Dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "model_family": MODEL_FAMILY,
            "feature_names": list(CANDIDATE_FEATURE_NAMES),
            "config": asdict(self.config),
            "state_dict": self.state_dict(),
        }
        if extra:
            payload.update(dict(extra))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, str(path))

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> "PosthostOneEditScorer":
        payload = torch.load(str(path), map_location=map_location)
        family = str(payload.get("model_family", "") or "")
        if family != MODEL_FAMILY:
            raise RuntimeError(f"Expected {MODEL_FAMILY}, got {family or 'unknown'}")
        config = dict(payload.get("config", {}) or {})
        model = cls(**config)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model


def masked_argmax(logits: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
    masked = logits.masked_fill(~candidate_mask.to(dtype=torch.bool), float("-inf"))
    return masked.argmax(dim=-1)


def feature_name_index(name: str) -> int:
    return int(CANDIDATE_FEATURE_NAMES.index(str(name)))


def feature_dim() -> int:
    return int(len(CANDIDATE_FEATURE_NAMES))


def ensure_feature_shape(row: Sequence[float]) -> list[float]:
    values = [float(x) for x in row]
    if len(values) != len(CANDIDATE_FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(CANDIDATE_FEATURE_NAMES)} candidate features, got {len(values)}"
        )
    return values
