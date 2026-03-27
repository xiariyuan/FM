from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DetTrackMatchLoss(nn.Module):
    """
    Detection-to-Track matching supervision (InfoNCE-style).

    This loss directly optimizes the association objective used at inference:
    given detection features at time t and track features from history (< t),
    enforce the correct track to have the highest similarity.

    Inputs follow the FM-Track/MOTIP tensor conventions:
      - trajectory_features: (B, G, T, N, C)
      - trajectory_id_labels: (B, G, T, N)  (constant per track across time)
      - trajectory_masks: (B, G, T, N)      (True = padded/invalid)
      - unknown_features: (B, G, T, M, C)
      - unknown_id_labels: (B, G, T, M)     (newborn label = num_id_vocabulary)
      - unknown_masks: (B, G, T, M)         (True = padded/invalid)
    """

    def __init__(
        self,
        temperature: float = 0.07,
        normalize: bool = True,
        causal: bool = True,
        newborn_label: Optional[int] = None,
    ):
        super().__init__()
        self.temperature = float(temperature)
        self.normalize = bool(normalize)
        self.causal = bool(causal)
        self.newborn_label = int(newborn_label) if newborn_label is not None else None

    def forward(
        self,
        unknown_features: torch.Tensor,
        unknown_id_labels: torch.Tensor,
        unknown_masks: torch.Tensor,
        trajectory_features: torch.Tensor,
        trajectory_id_labels: torch.Tensor,
        trajectory_masks: torch.Tensor,
        newborn_label: Optional[int] = None,
    ) -> torch.Tensor:
        device = unknown_features.device

        if newborn_label is None:
            newborn_label = self.newborn_label
        newborn_label = int(newborn_label) if newborn_label is not None else None

        if unknown_features.dim() != 5 or trajectory_features.dim() != 5:
            raise ValueError(
                "DetTrackMatchLoss expects 5D tensors: "
                f"unknown_features={tuple(unknown_features.shape)}, "
                f"trajectory_features={tuple(trajectory_features.shape)}"
            )
        if unknown_id_labels.dim() != 4 or trajectory_id_labels.dim() != 4:
            raise ValueError(
                "DetTrackMatchLoss expects 4D label tensors: "
                f"unknown_id_labels={tuple(unknown_id_labels.shape)}, "
                f"trajectory_id_labels={tuple(trajectory_id_labels.shape)}"
            )
        if unknown_masks.dim() != 4 or trajectory_masks.dim() != 4:
            raise ValueError(
                "DetTrackMatchLoss expects 4D mask tensors: "
                f"unknown_masks={tuple(unknown_masks.shape)}, "
                f"trajectory_masks={tuple(trajectory_masks.shape)}"
            )

        # Ensure boolean masks
        if unknown_masks.dtype != torch.bool:
            unknown_masks = unknown_masks.to(torch.bool)
        if trajectory_masks.dtype != torch.bool:
            trajectory_masks = trajectory_masks.to(torch.bool)

        B, G, T, N, C = trajectory_features.shape
        _, _, Tu, M, Cu = unknown_features.shape
        if Tu != T:
            raise ValueError(f"Time dim mismatch: trajectory T={T} vs unknown T={Tu}")
        if Cu != C:
            raise ValueError(f"Feature dim mismatch: trajectory C={C} vs unknown C={Cu}")

        # Track labels are constant per track, but tracks may be padded at early frames (start later).
        # Use the first *valid* timestep per track to avoid picking a masked (-1) label at t=0.
        if trajectory_masks is not None:
            valid_traj_any = (~trajectory_masks).any(dim=2)  # (B, G, N)
            first_valid_t = (~trajectory_masks).float().argmax(dim=2)  # (B, G, N)
            gather_idx = first_valid_t.unsqueeze(2)  # (B, G, 1, N)
            track_labels = torch.gather(trajectory_id_labels, dim=2, index=gather_idx).squeeze(2)  # (B, G, N)
            track_labels = track_labels.masked_fill(~valid_traj_any, -1)
        else:
            track_labels = trajectory_id_labels[:, :, 0, :]  # (B, G, N)

        # Build history prototypes for each track at each time t:
        # prototype[t, n] = last valid trajectory feature for track n in frames < t (or <= t if not causal).
        valid_traj = ~trajectory_masks  # (B, G, T, N)
        if self.causal:
            valid_hist = torch.cat(
                [torch.zeros_like(valid_traj[:, :, :1, :]), valid_traj[:, :, :-1, :]],
                dim=2,
            )
        else:
            valid_hist = valid_traj

        idx = torch.arange(T, device=device, dtype=torch.long).view(1, 1, T, 1).expand(B, G, T, N)
        masked_idx = torch.where(valid_hist, idx, torch.full_like(idx, -1))
        last_idx = torch.cummax(masked_idx, dim=2).values  # (B, G, T, N) in [-1, T-1]
        proto_valid = last_idx >= 0

        gather_idx = last_idx.clamp(min=0).unsqueeze(-1).expand(B, G, T, N, C)
        proto_feats = torch.gather(trajectory_features, dim=2, index=gather_idx)
        proto_feats = proto_feats.masked_fill(~proto_valid.unsqueeze(-1), 0.0)

        # NOTE: Do not normalize here under autocast; padded prototypes are all-zero vectors and
        # float16 default eps=1e-12 can underflow, producing NaNs (0/0). We normalize later in float32.

        tau = max(self.temperature, 1e-6)

        # Vectorize over (B*G) and process per time step to avoid slow Python triply-nested loops.
        BG = B * G
        track_labels = track_labels.reshape(BG, N)
        proto_feats = proto_feats.reshape(BG, T, N, C)
        proto_valid = proto_valid.reshape(BG, T, N)

        unk_feats = unknown_features.reshape(BG, T, M, C)
        unk_masks = unknown_masks.reshape(BG, T, M)
        unk_labels = unknown_id_labels.reshape(BG, T, M)

        total_loss = unknown_features.sum() * 0.0
        total_count = 0

        # Precompute static track validity based on labels (and optional newborn exclusion).
        static_track_valid = track_labels >= 0
        if newborn_label is not None:
            static_track_valid = static_track_valid & (track_labels != newborn_label)

        for t in range(T):
            key_valid = proto_valid[:, t] & static_track_valid  # (BG, N)
            key_count = key_valid.sum(dim=-1)  # (BG,)
            group_keep = key_count >= 2
            if not bool(group_keep.any()):
                continue

            gv = group_keep.nonzero(as_tuple=True)[0]

            keys = proto_feats[gv, t]  # (Gv, N, C)
            queries = unk_feats[gv, t]  # (Gv, M, C)

            # Prefer float32 for stability of log_softmax under autocast.
            if keys.dtype in (torch.float16, torch.bfloat16):
                keys = keys.float()
            if queries.dtype in (torch.float16, torch.bfloat16):
                queries = queries.float()

            if self.normalize:
                keys = F.normalize(keys, p=2, dim=-1)
                queries = F.normalize(queries, p=2, dim=-1)

            logits = torch.einsum("bmc,bnc->bmn", queries, keys) / tau  # (Gv, M, N)
            logits = logits.masked_fill(~key_valid[gv].unsqueeze(1), float("-inf"))

            q_labels = unk_labels[gv, t]  # (Gv, M)
            q_valid = (~unk_masks[gv, t]) & (q_labels >= 0)
            if newborn_label is not None:
                q_valid = q_valid & (q_labels != newborn_label)

            # Multi-positive friendly: allow multiple matching keys if vocab collisions ever happen.
            eq = q_labels.unsqueeze(-1).eq(track_labels[gv].unsqueeze(1))  # (Gv, M, N)
            pos_mask = eq & key_valid[gv].unsqueeze(1)
            pos_count = pos_mask.sum(dim=-1)  # (Gv, M)

            q_keep = q_valid & (pos_count > 0)
            if not bool(q_keep.any()):
                continue

            log_probs = logits.log_softmax(dim=-1)
            # Avoid (-inf)*0 => NaN by zeroing non-positive entries before summing.
            pos_log_probs = log_probs.masked_fill(~pos_mask, 0.0)
            loss_per_q = -pos_log_probs.sum(dim=-1) / pos_count.clamp(min=1).to(pos_log_probs.dtype)

            total_loss = total_loss + loss_per_q.masked_select(q_keep).sum()
            total_count += int(q_keep.sum().item())

        if total_count <= 0:
            return unknown_features.sum() * 0.0
        return total_loss / float(total_count)


def build_det_track_match_loss(config: dict) -> DetTrackMatchLoss:
    num_id_vocabulary = config.get("NUM_ID_VOCABULARY", None)
    newborn_label = None
    if num_id_vocabulary is not None:
        try:
            newborn_label = int(num_id_vocabulary)
        except Exception:
            newborn_label = None

    return DetTrackMatchLoss(
        temperature=float(config.get("DET_TRACK_MATCH_TEMPERATURE", 0.07)),
        normalize=bool(config.get("DET_TRACK_MATCH_NORMALIZE", True)),
        causal=bool(config.get("DET_TRACK_MATCH_CAUSAL", True)),
        newborn_label=newborn_label,
    )
