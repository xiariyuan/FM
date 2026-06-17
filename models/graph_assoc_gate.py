from __future__ import annotations

import torch
from torch import nn


def _make_gate_backbone(input_dim: int, hidden_dim: int, dropout: float, num_hidden_layers: int) -> nn.Module:
    if int(hidden_dim) <= 0:
        return nn.Identity()

    depth = max(1, int(num_hidden_layers))
    layers: list[nn.Module] = [
        nn.Linear(int(input_dim), int(hidden_dim)),
        nn.ReLU(inplace=True),
        nn.Dropout(p=float(dropout)),
    ]
    for _ in range(depth - 1):
        layers.extend(
            [
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                nn.ReLU(inplace=True),
                nn.Dropout(p=float(dropout)),
            ]
        )
    return nn.Sequential(*layers)


class GraphAssocGate(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        dropout: float = 0.0,
        num_hidden_layers: int = 1,
    ) -> None:
        super().__init__()
        input_dim = int(input_dim)
        hidden_dim = int(hidden_dim)
        dropout = float(dropout)
        self.num_hidden_layers = max(1, int(num_hidden_layers))
        if hidden_dim <= 0:
            self.net = nn.Linear(input_dim, 1)
        else:
            self.net = nn.Sequential(
                _make_gate_backbone(input_dim, hidden_dim, dropout, self.num_hidden_layers),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class GraphAssocDualHeadGate(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        dropout: float = 0.0,
        num_hidden_layers: int = 1,
    ) -> None:
        super().__init__()
        input_dim = int(input_dim)
        hidden_dim = int(hidden_dim)
        dropout = float(dropout)
        self.num_hidden_layers = max(1, int(num_hidden_layers))
        if hidden_dim <= 0:
            self.backbone = nn.Identity()
            head_input_dim = input_dim
        else:
            self.backbone = _make_gate_backbone(input_dim, hidden_dim, dropout, self.num_hidden_layers)
            head_input_dim = hidden_dim
        self.gain_head = nn.Linear(head_input_dim, 1)
        self.neutral_head = nn.Linear(head_input_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(x)
        gain_logit = self.gain_head(feat).squeeze(-1)
        neutral_logit = self.neutral_head(feat).squeeze(-1)
        return gain_logit, neutral_logit
