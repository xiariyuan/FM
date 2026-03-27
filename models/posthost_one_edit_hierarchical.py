from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn


MODEL_FAMILY = "posthost_one_edit_hierarchical_v1"


@dataclass
class BinaryMLPConfig:
    input_dim: int
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.1


class BinaryMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.config = BinaryMLPConfig(
            input_dim=int(input_dim),
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            dropout=float(dropout),
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
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.to(dtype=torch.float32)).squeeze(-1).clamp(min=-30.0, max=30.0)


@dataclass
class CandidateRankerConfig:
    input_dim: int
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.1


class CandidateRanker(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.config = CandidateRankerConfig(
            input_dim=int(input_dim),
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            dropout=float(dropout),
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
        self.net = nn.Sequential(*layers)

    def forward(self, candidate_features: torch.Tensor) -> torch.Tensor:
        return self.net(candidate_features.to(dtype=torch.float32)).squeeze(-1).clamp(min=-30.0, max=30.0)


class PosthostOneEditHierarchical(nn.Module):
    def __init__(
        self,
        cluster_feature_dim: int,
        candidate_feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.cluster_feature_dim = int(cluster_feature_dim)
        self.candidate_feature_dim = int(candidate_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)

        self.keep_edit_gate = BinaryMLP(
            input_dim=self.cluster_feature_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        self.defer_swap_selector = BinaryMLP(
            input_dim=self.cluster_feature_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        self.defer_ranker = CandidateRanker(
            input_dim=self.candidate_feature_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        self.swap_ranker = CandidateRanker(
            input_dim=self.candidate_feature_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        extra: Dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "model_family": MODEL_FAMILY,
            "cluster_feature_dim": int(self.cluster_feature_dim),
            "candidate_feature_dim": int(self.candidate_feature_dim),
            "hidden_dim": int(self.hidden_dim),
            "num_layers": int(self.num_layers),
            "dropout": float(self.dropout),
            "keep_edit_gate_config": asdict(self.keep_edit_gate.config),
            "defer_swap_selector_config": asdict(self.defer_swap_selector.config),
            "defer_ranker_config": asdict(self.defer_ranker.config),
            "swap_ranker_config": asdict(self.swap_ranker.config),
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
    ) -> "PosthostOneEditHierarchical":
        payload = torch.load(str(path), map_location=map_location)
        family = str(payload.get("model_family", "") or "")
        if family != MODEL_FAMILY:
            raise RuntimeError(f"Expected {MODEL_FAMILY}, got {family or 'unknown'}")
        model = cls(
            cluster_feature_dim=int(payload["cluster_feature_dim"]),
            candidate_feature_dim=int(payload["candidate_feature_dim"]),
            hidden_dim=int(payload.get("hidden_dim", 128)),
            num_layers=int(payload.get("num_layers", 3)),
            dropout=float(payload.get("dropout", 0.1)),
        )
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model
