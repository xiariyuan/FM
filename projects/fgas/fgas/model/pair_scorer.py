from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch import nn


MODEL_FAMILY = "fgas_pair_scorer"


class FGASPairScorer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        input_dim = int(input_dim)
        hidden_dim = int(hidden_dim)
        dropout = float(dropout)
        if hidden_dim <= 0:
            self.net = nn.Linear(input_dim, 1)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class PairGroupPrediction:
    group_key: str
    logits: List[float]
    probs: List[float]
    labels: List[int]
    ambiguous: bool
