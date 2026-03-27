from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .laplace_signature import build_laplace_track_prototypes
from .reliability_head import LaplaceReliabilityHead


class LaplaceAssociationAdapter(nn.Module):
    def __init__(
        self,
        decay_scales: Iterable[float] = (1.0, 2.0, 4.0),
        hidden_dim: int = 16,
        blend: float = 0.35,
    ):
        super().__init__()
        self.decay_scales = tuple(float(x) for x in decay_scales)
        self.head = LaplaceReliabilityHead(pair_dim=6, hidden_dim=hidden_dim)
        self.blend = float(blend)

    def forward(
        self,
        spatial_scores: torch.Tensor,
        det_features: torch.Tensor,
        track_history_features: torch.Tensor,
        track_history_masks: torch.Tensor,
        motion_scores: Optional[torch.Tensor] = None,
        det_scores: Optional[torch.Tensor] = None,
    ) -> dict:
        if spatial_scores.numel() == 0:
            return {
                "fused_scores": spatial_scores,
                "laplace_scores": spatial_scores,
                "weights": None,
                "reliability": None,
                "stability": None,
                "coherence": None,
            }

        device = spatial_scores.device
        score_dtype = spatial_scores.dtype
        work_dtype = torch.float32 if score_dtype in (torch.float16, torch.bfloat16) else score_dtype

        spatial = spatial_scores.to(dtype=work_dtype)
        det = det_features.to(device=device, dtype=work_dtype)
        history = track_history_features.to(device=device, dtype=work_dtype)
        masks = track_history_masks.to(device=device)

        if motion_scores is None:
            motion = spatial
        else:
            motion = motion_scores.to(device=device, dtype=work_dtype)

        if det_scores is None:
            det_conf = torch.ones((spatial.shape[0], 1), device=device, dtype=work_dtype)
        else:
            det_conf = det_scores.to(device=device, dtype=work_dtype).view(-1, 1).clamp(min=0.0, max=1.0)

        prototypes, aux = build_laplace_track_prototypes(
            track_features=history,
            track_masks=masks,
            decay_scales=self.decay_scales,
        )
        det_norm = F.normalize(det, dim=-1)
        proto_norm = F.normalize(prototypes, dim=-1)
        laplace_multi = torch.einsum("nd,mkd->nmk", det_norm, proto_norm)
        laplace_scores = ((laplace_multi.clamp(min=-1.0, max=1.0) + 1.0) * 0.5).mean(dim=-1)

        agreement = 1.0 - (spatial - laplace_scores).abs()
        agreement = agreement.clamp(min=0.0, max=1.0)
        stability = aux["stability"].view(1, -1).expand_as(spatial)
        coherence = aux["coherence"].view(1, -1).expand_as(spatial)
        det_conf_full = det_conf.expand(-1, spatial.shape[1])

        pair_features = torch.stack(
            [spatial, laplace_scores, motion, agreement, stability, det_conf_full * coherence],
            dim=-1,
        )
        component_scores = torch.stack([spatial, laplace_scores, motion], dim=-1)
        head_out = self.head(pair_features, component_scores)
        fused = (1.0 - self.blend) * spatial + self.blend * head_out["fused_scores"]
        return {
            "fused_scores": fused.to(dtype=score_dtype).clamp(min=0.0, max=1.0),
            "laplace_scores": laplace_scores.to(dtype=score_dtype),
            "weights": head_out["weights"],
            "reliability": head_out["reliability"],
            "stability": aux["stability"].to(dtype=score_dtype),
            "coherence": aux["coherence"].to(dtype=score_dtype),
        }
