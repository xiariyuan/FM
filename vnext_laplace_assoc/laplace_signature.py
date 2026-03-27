from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn.functional as F


def _to_scale_tensor(
    decay_scales: Iterable[float],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    values = [float(x) for x in decay_scales]
    if len(values) == 0:
        values = [1.0, 2.0, 4.0]
    return torch.tensor(values, device=device, dtype=dtype).clamp(min=1e-3)


def _last_valid_indices(track_masks: torch.Tensor) -> torch.Tensor:
    valid = (~track_masks).to(dtype=torch.long)
    time_idx = torch.arange(track_masks.shape[1], device=track_masks.device, dtype=torch.long).view(1, -1)
    last = (valid * (time_idx + 1)).max(dim=1).values - 1
    return last.clamp(min=0)


def gather_last_valid(track_features: torch.Tensor, track_masks: torch.Tensor) -> torch.Tensor:
    if track_features.numel() == 0:
        return torch.zeros(
            (track_features.shape[0], track_features.shape[-1]),
            device=track_features.device,
            dtype=track_features.dtype,
        )
    last = _last_valid_indices(track_masks)
    gather_index = last.view(-1, 1, 1).expand(-1, 1, track_features.shape[-1])
    return torch.gather(track_features, dim=1, index=gather_index).squeeze(1)


def build_laplace_track_prototypes(
    track_features: torch.Tensor,
    track_masks: torch.Tensor,
    decay_scales: Iterable[float],
) -> Tuple[torch.Tensor, dict]:
    if track_features.dim() != 3:
        raise ValueError(f"Expected track_features (M,T,D), got {tuple(track_features.shape)}")
    if track_masks.dim() != 2:
        raise ValueError(f"Expected track_masks (M,T), got {tuple(track_masks.shape)}")

    device = track_features.device
    work_dtype = torch.float32 if track_features.dtype in (torch.float16, torch.bfloat16) else track_features.dtype
    feats = track_features.to(dtype=work_dtype)
    masks = track_masks.to(device=device)

    num_tracks, window, _ = feats.shape
    scales = _to_scale_tensor(decay_scales, device=device, dtype=work_dtype)
    num_scales = int(scales.numel())

    if num_tracks == 0 or window == 0:
        empty = torch.zeros((num_tracks, num_scales, feats.shape[-1]), device=device, dtype=work_dtype)
        return empty, {
            "stability": torch.zeros((num_tracks,), device=device, dtype=work_dtype),
            "coherence": torch.zeros((num_tracks,), device=device, dtype=work_dtype),
            "last_features": torch.zeros((num_tracks, feats.shape[-1]), device=device, dtype=work_dtype),
        }

    valid = (~masks).to(dtype=work_dtype)
    last_idx = _last_valid_indices(masks)
    time_idx = torch.arange(window, device=device, dtype=work_dtype).view(1, window)
    age = last_idx.to(dtype=work_dtype).view(-1, 1) - time_idx
    age = age.clamp(min=0.0)

    basis = torch.exp(-age.unsqueeze(1) / scales.view(1, num_scales, 1))
    basis = basis * valid.unsqueeze(1)
    norm = basis.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    basis = basis / norm

    prototypes = torch.einsum("mkt,mtd->mkd", basis, feats)
    last_features = gather_last_valid(feats, masks)
    last_norm = F.normalize(last_features, dim=-1)
    proto_norm = F.normalize(prototypes, dim=-1)
    coherence = ((proto_norm * last_norm.unsqueeze(1)).sum(dim=-1).clamp(min=-1.0, max=1.0) + 1.0) * 0.5
    coherence = coherence.mean(dim=1)

    if window >= 3:
        valid_triplet = valid[:, 2:] * valid[:, 1:-1] * valid[:, :-2]
        delta2 = feats[:, 2:] - 2.0 * feats[:, 1:-1] + feats[:, :-2]
        curvature = delta2.pow(2).mean(dim=-1).sqrt()
        curvature = (curvature * valid_triplet).sum(dim=1) / valid_triplet.sum(dim=1).clamp(min=1.0)
    else:
        curvature = torch.zeros((num_tracks,), device=device, dtype=work_dtype)

    stability = torch.exp(-curvature).clamp(min=0.0, max=1.0)
    return prototypes, {
        "stability": stability,
        "coherence": coherence,
        "last_features": last_features,
    }
