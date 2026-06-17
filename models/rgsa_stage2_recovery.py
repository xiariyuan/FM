"""Stage 2 local recovery head for RGSA.

For each Stage-1-deferred detection, re-scores its top-k track candidates
using HACA pair features and predicts rewrite/defer/reject.

Architecture:
  - Per-candidate MLP (15 -> 32 -> 16 -> 1) scores each candidate
  - Global action head (aggregated_features -> 3) predicts rewrite/defer/reject
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rgsa_contract import (
    HACA_PAIR_FEATURE_DIM,
    HACA_PAIR_FEATURE_NAMES,
    Stage2Output,
)


class Stage2RecoveryHead(nn.Module):
    """Per-candidate scoring + action classification."""

    def __init__(
        self,
        pair_dim: int = HACA_PAIR_FEATURE_DIM,
        hidden_dim: int = 32,
        action_hidden: int = 16,
        num_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Per-candidate scorer
        self.candidate_scorer = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Action head: takes aggregated candidate features + score stats
        # Input: pair_dim (mean of top-k features) + 3 (max_score, mean_score, score_std)
        action_input_dim = pair_dim + 3
        self.action_head = nn.Sequential(
            nn.Linear(action_input_dim, action_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(action_hidden, num_classes),
        )

    def forward(
        self,
        candidate_features: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Args:
            candidate_features: (N, K, pair_dim) per-candidate features
            candidate_mask: (N, K) bool mask, True for valid candidates

        Returns:
            candidate_scores: (N, K) per-candidate scores
            action_logits: (N, 3) rewrite/defer/reject logits
        """
        N, K, D = candidate_features.shape
        flat = candidate_features.reshape(N * K, D)
        scores_flat = self.candidate_scorer(flat).reshape(N, K)

        if candidate_mask is not None:
            scores_flat = scores_flat.masked_fill(~candidate_mask, -1e9)

        # Aggregated features for action head
        if candidate_mask is not None:
            valid_counts = candidate_mask.sum(dim=1, keepdim=True).clamp(min=1)
            mask_f = candidate_mask.unsqueeze(-1).float()
            mean_feat = (candidate_features * mask_f).sum(1) / valid_counts
            # For score aggregation, replace masked positions with 0 (not -inf)
            safe_scores = scores_flat.clone()
            safe_scores[~candidate_mask] = 0.0
            valid_count_per_row = candidate_mask.sum(dim=1).clamp(min=1).float()
            mean_score = safe_scores.sum(dim=1) / valid_count_per_row
            max_score = scores_flat.masked_fill(~candidate_mask, -1e9).max(dim=1).values
            # std with safe fallback for single-element groups
            score_var = (safe_scores.pow(2).sum(dim=1) / valid_count_per_row) - mean_score.pow(2)
            score_std = score_var.clamp(min=0).sqrt()
        else:
            mean_feat = candidate_features.mean(dim=1)
            max_score = scores_flat.max(dim=1).values
            mean_score = scores_flat.mean(dim=1)
            score_std = scores_flat.std(dim=1)

        agg = torch.cat([mean_feat, max_score.unsqueeze(-1), mean_score.unsqueeze(-1), score_std.unsqueeze(-1)], dim=-1)
        action_logits = self.action_head(agg)

        return scores_flat, action_logits

    @torch.no_grad()
    def predict(
        self,
        candidate_features: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> tuple:
        """Returns (candidate_scores, action_indices, action_probs)."""
        scores, action_logits = self.forward(candidate_features, candidate_mask)
        probs = F.softmax(action_logits, dim=-1)
        actions = probs.argmax(dim=-1)
        return scores, actions, probs

    @torch.no_grad()
    def apply_recovery(
        self,
        deferred_det_ids: List[int],
        candidate_features_per_det: Dict[int, np.ndarray],
        candidate_track_ids_per_det: Dict[int, List[int]],
        device: str = "cpu",
    ) -> Stage2Output:
        """Run inference on deferred detections.

        Args:
            deferred_det_ids: detection ids deferred from Stage 1
            candidate_features_per_det: det_id -> (K, pair_dim) features
            candidate_track_ids_per_det: det_id -> [track_ids]

        Returns:
            Stage2Output
        """
        output = Stage2Output()
        if not deferred_det_ids:
            return output

        # Batch all deferred detections
        all_feats = []
        all_masks = []
        max_k = 0
        for det_id in deferred_det_ids:
            feats = candidate_features_per_det.get(det_id)
            if feats is None or len(feats) == 0:
                output.rejected_det_ids.append(det_id)
                continue
            max_k = max(max_k, len(feats))
            all_feats.append(feats)
        if not all_feats:
            return output

        # Pad to max_k
        padded = []
        masks = []
        for feats in all_feats:
            k = len(feats)
            pad = np.zeros((max_k - k, feats.shape[1]), dtype=np.float32)
            padded.append(np.concatenate([feats, pad], axis=0))
            mask = np.zeros((max_k,), dtype=bool)
            mask[:k] = True
            masks.append(mask)

        x = torch.tensor(np.stack(padded), dtype=torch.float32, device=device)
        m = torch.tensor(np.stack(masks), dtype=torch.bool, device=device)

        scores, actions, probs = self.predict(x, m)

        valid_idx = 0
        for det_id in deferred_det_ids:
            if det_id in output.rejected_det_ids:
                continue
            a = int(actions[valid_idx])
            p = probs[valid_idx]
            track_ids = candidate_track_ids_per_det.get(det_id, [])
            best_score = float(scores[valid_idx].max())

            if a == 0 and track_ids:  # rewrite
                best_cand = int(scores[valid_idx].argmax())
                if best_cand < len(track_ids):
                    output.rewritten_matches[det_id] = track_ids[best_cand]
                    output.best_recovery_scores[det_id] = best_score
                else:
                    output.rejected_det_ids.append(det_id)
            elif a == 1:  # defer (send to Stage 3)
                output.still_deferred_det_ids.append(det_id)
                output.best_recovery_scores[det_id] = best_score
            else:  # reject
                output.rejected_det_ids.append(det_id)

            valid_idx += 1

        return output

    def save_checkpoint(self, path: str, metadata: Optional[Dict] = None):
        ckpt = {
            "state_dict": self.state_dict(),
            "pair_dim": HACA_PAIR_FEATURE_DIM,
            "metadata": metadata or {},
        }
        torch.save(ckpt, path)

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> "Stage2RecoveryHead":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = cls()
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        return model
