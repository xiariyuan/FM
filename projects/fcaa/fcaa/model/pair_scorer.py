from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import torch
from torch import nn


MODEL_FAMILY = "fcaa_pair_scorer"

CONTROL_FEATURE_NAMES = ["s_reid"]
FCAA_FEATURE_NAMES = ["s_reid", "s_low", "s_mid", "s_high"]


class FCAAPairScorer(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(int(input_dim), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x).squeeze(-1)
        return logits


@dataclass
class GroupPrediction:
    group_key: str
    logits: List[float]
    probs: List[float]
    labels: List[int]
    ambiguous: bool


def feature_names(mode: str) -> List[str]:
    mode_key = str(mode).lower()
    if mode_key == "control":
        return list(CONTROL_FEATURE_NAMES)
    if mode_key == "freq":
        return list(FCAA_FEATURE_NAMES)
    raise ValueError(f"Unsupported feature mode: {mode}")


def row_to_feature(row: Dict[str, object], mode: str) -> List[float]:
    names = feature_names(mode)
    return [float(row[name]) for name in names]


def sigmoid_probabilities(logits: Iterable[float]) -> List[float]:
    tensor = torch.as_tensor(list(logits), dtype=torch.float32)
    return torch.sigmoid(tensor).tolist()
