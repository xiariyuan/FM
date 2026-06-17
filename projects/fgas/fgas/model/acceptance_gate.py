from __future__ import annotations

import torch
from torch import nn


class FGASAcceptanceGate(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.0) -> None:
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
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
