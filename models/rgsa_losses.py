"""Shared loss functions for RGSA stage heads."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.Module):
    """Cross-entropy with per-class weights and optional focal modulation."""

    def __init__(
        self,
        class_weights: Optional[list] = None,
        focal_gamma: float = 0.0,
    ):
        super().__init__()
        weight = None
        if class_weights is not None:
            weight = torch.tensor(class_weights, dtype=torch.float32)
        self.register_buffer("weight", weight)
        self.focal_gamma = focal_gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (N, C) raw logits
            targets: (N,) integer class labels
        """
        if self.focal_gamma > 0:
            probs = F.softmax(logits, dim=-1)
            ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            focal = (1.0 - pt) ** self.focal_gamma
            return (focal * ce).mean()
        return F.cross_entropy(logits, targets, weight=self.weight)


class StagePairLoss(nn.Module):
    """Loss for Stage 2: per-candidate scoring + action classification.

    Two-head design:
      - score_head: scores each candidate (regression to GT rank)
      - action_head: 3-class (rewrite/defer/reject)
    """

    def __init__(
        self,
        action_class_weights: Optional[list] = None,
        score_loss_weight: float = 1.0,
        action_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.action_ce = WeightedCrossEntropyLoss(class_weights=action_class_weights)
        self.score_loss_weight = score_loss_weight
        self.action_loss_weight = action_loss_weight

    def forward(
        self,
        candidate_scores: torch.Tensor,
        action_logits: torch.Tensor,
        target_rank: torch.Tensor,
        target_action: torch.Tensor,
    ) -> dict:
        """
        Args:
            candidate_scores: (N, K) per-candidate scores
            action_logits: (N, 3) action logits
            target_rank: (N,) index of correct candidate (or -1 if none)
            target_action: (N,) action label (0=rewrite, 1=defer, 2=reject)
        """
        # Score loss: cross-entropy over candidate ranking
        valid = target_rank >= 0
        score_loss = torch.tensor(0.0, device=candidate_scores.device)
        if valid.any():
            score_loss = F.cross_entropy(
                candidate_scores[valid], target_rank[valid]
            )

        # Action loss
        action_loss = self.action_ce(action_logits, target_action)

        total = self.score_loss_weight * score_loss + self.action_loss_weight * action_loss
        return {
            "total": total,
            "score_loss": score_loss.detach(),
            "action_loss": action_loss.detach(),
        }
