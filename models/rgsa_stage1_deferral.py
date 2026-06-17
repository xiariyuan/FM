"""Stage 1 deferral head for RGSA.

Lightweight MLP that predicts accept/defer/reject for each det-track pair
based on low-dimensional HACA runtime features.

Input (11 dims):
  activation, margin, entropy, bg_prob,
  beta_hist, beta_ood, ood_score,
  track_gap, track_age, history_len, det_score

Output (3 logits): accept / defer / reject
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rgsa_contract import STAGE1_FEATURE_DIM, STAGE1_FEATURE_NAMES, Stage1Output


class Stage1DeferralHead(nn.Module):
    """MLP accept/defer/reject classifier."""

    def __init__(
        self,
        input_dim: int = STAGE1_FEATURE_DIM,
        hidden_dims: tuple = (32, 16),
        num_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, input_dim) -> (N, 3) logits"""
        return self.net(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple:
        """Returns (action_indices, action_probs)."""
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        actions = probs.argmax(dim=-1)
        return actions, probs

    @torch.no_grad()
    def apply_soft_deferral(
        self,
        features: np.ndarray,
        det_ids: list,
        track_ids: list,
        device: str = "cpu",
        lambda_defer: float = 0.3,
        lambda_reject: float = 0.8,
    ) -> Stage1Output:
        """Run inference and produce Stage1Output with soft cost biases.

        Args:
            features: (N, 11) array of per-pair HACA features
            det_ids: list of detection ids
            track_ids: list of track ids (same length as det_ids)
            device: torch device
            lambda_defer: cost penalty multiplier for defer
            lambda_reject: cost penalty multiplier for reject

        Returns:
            Stage1Output with accepted/deferred/rejected split
        """
        x = torch.tensor(features, dtype=torch.float32, device=device)
        actions, probs = self.predict(x)

        output = Stage1Output()
        for i, (det_id, track_id) in enumerate(zip(det_ids, track_ids)):
            a = int(actions[i])
            p = probs[i]
            if a == 0:  # accept
                output.accepted_det_ids.append(det_id)
                output.accepted_matches[det_id] = track_id
            elif a == 1:  # defer
                output.deferred_det_ids.append(det_id)
                output.deferred_cost_bias[det_id] = lambda_defer * float(1.0 - p[0])
                output.deferred_host_signals[det_id] = {
                    "p_accept": float(p[0]),
                    "p_defer": float(p[1]),
                    "p_reject": float(p[2]),
                }
            else:  # reject
                output.rejected_det_ids.append(det_id)
                output.deferred_cost_bias[det_id] = lambda_reject

        return output

    def save_checkpoint(self, path: str, metadata: Optional[Dict] = None):
        ckpt = {
            "state_dict": self.state_dict(),
            "input_dim": STAGE1_FEATURE_DIM,
            "metadata": metadata or {},
        }
        torch.save(ckpt, path)

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> "Stage1DeferralHead":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = cls()
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        return model
