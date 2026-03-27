from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def _last_valid_indices(track_masks: torch.Tensor) -> torch.Tensor:
    valid = (~track_masks).to(dtype=torch.long)
    time_idx = torch.arange(track_masks.shape[1], device=track_masks.device, dtype=torch.long).view(1, -1)
    last = (valid * (time_idx + 1)).max(dim=1).values - 1
    return last.clamp(min=0)


def _bbox_distance_scores(
    det_boxes_cxcywh: torch.Tensor,
    track_boxes_cxcywh: torch.Tensor,
    tau: float = 1.0,
) -> torch.Tensor:
    if det_boxes_cxcywh.numel() == 0 or track_boxes_cxcywh.numel() == 0:
        return torch.zeros(
            (det_boxes_cxcywh.shape[0], track_boxes_cxcywh.shape[0]),
            device=det_boxes_cxcywh.device,
            dtype=det_boxes_cxcywh.dtype,
        )

    det = det_boxes_cxcywh.to(dtype=torch.float32)
    trk = track_boxes_cxcywh.to(dtype=torch.float32)
    eps = 1e-6

    det_cx = det[:, 0].unsqueeze(1)
    det_cy = det[:, 1].unsqueeze(1)
    det_w = det[:, 2].unsqueeze(1).clamp(min=eps)
    det_h = det[:, 3].unsqueeze(1).clamp(min=eps)

    trk_cx = trk[:, 0].unsqueeze(0)
    trk_cy = trk[:, 1].unsqueeze(0)
    trk_w = trk[:, 2].unsqueeze(0).clamp(min=eps)
    trk_h = trk[:, 3].unsqueeze(0).clamp(min=eps)

    dx = (det_cx - trk_cx) / trk_w
    dy = (det_cy - trk_cy) / trk_h
    dw = torch.log(det_w / trk_w)
    dh = torch.log(det_h / trk_h)

    dist = torch.sqrt(dx * dx + dy * dy + dw * dw + dh * dh + eps)
    tau = max(float(tau), eps)
    sim = torch.exp(-dist / tau)
    return sim.to(device=det_boxes_cxcywh.device, dtype=det_boxes_cxcywh.dtype).clamp(min=0.0, max=1.0)


def compute_laplace_supervision_loss(
    *,
    seq_info: Dict,
    laplace_assoc,
    num_id_vocabulary: int,
    temperature: float = 1.0,
    background_weight: float = 0.25,
    bbox_tau: float = 1.0,
    min_history: int = 1,
) -> Dict[str, torch.Tensor]:
    traj_feats = seq_info.get("trajectory_features", None)
    traj_boxes = seq_info.get("trajectory_boxes", None)
    traj_labels = seq_info.get("trajectory_id_labels", None)
    traj_masks = seq_info.get("trajectory_masks", None)
    unk_feats = seq_info.get("unknown_features", None)
    unk_boxes = seq_info.get("unknown_boxes", None)
    unk_labels = seq_info.get("unknown_id_labels", None)
    unk_masks = seq_info.get("unknown_masks", None)

    if (
        traj_feats is None
        or traj_boxes is None
        or traj_labels is None
        or traj_masks is None
        or unk_feats is None
        or unk_boxes is None
        or unk_labels is None
        or unk_masks is None
    ):
        device = traj_feats.device if torch.is_tensor(traj_feats) else torch.device("cpu")
        zero = torch.tensor(0.0, device=device)
        return {
            "loss": zero,
            "matched_loss": zero,
            "background_loss": zero,
            "matched_rows": zero,
            "background_rows": zero,
            "active_tracks": zero,
            "frames_used": zero,
        }

    device = traj_feats.device
    work_dtype = torch.float32 if traj_feats.dtype in (torch.float16, torch.bfloat16) else traj_feats.dtype
    zero = torch.tensor(0.0, device=device, dtype=work_dtype)

    _, _, total_frames, _, _ = traj_feats.shape
    temp = max(float(temperature), 1e-4)
    min_history = max(int(min_history), 1)
    newborn_label = int(num_id_vocabulary)

    matched_terms = []
    background_terms = []
    matched_rows = 0
    background_rows = 0
    active_tracks_total = 0
    frames_used = 0

    for batch_idx in range(traj_feats.shape[0]):
        for group_idx in range(traj_feats.shape[1]):
            for frame_idx in range(min_history, total_frames):
                det_valid = (~unk_masks[batch_idx, group_idx, frame_idx]) & (unk_labels[batch_idx, group_idx, frame_idx] >= 0)
                if not det_valid.any():
                    continue

                hist_feats_all = traj_feats[batch_idx, group_idx, :frame_idx].permute(1, 0, 2).contiguous()
                hist_boxes_all = traj_boxes[batch_idx, group_idx, :frame_idx].permute(1, 0, 2).contiguous()
                hist_labels_all = traj_labels[batch_idx, group_idx, :frame_idx].permute(1, 0).contiguous()
                hist_masks_all = traj_masks[batch_idx, group_idx, :frame_idx].permute(1, 0).contiguous()

                active_track = (~hist_masks_all).any(dim=1)
                if not active_track.any():
                    continue

                hist_feats = hist_feats_all[active_track]
                hist_boxes = hist_boxes_all[active_track]
                hist_labels = hist_labels_all[active_track]
                hist_masks = hist_masks_all[active_track]

                last_idx = _last_valid_indices(hist_masks)
                row_idx = torch.arange(hist_feats.shape[0], device=device)
                last_feats = hist_feats[row_idx, last_idx]
                last_boxes = hist_boxes[row_idx, last_idx]
                last_labels = hist_labels[row_idx, last_idx]

                valid_track = (last_labels >= 0) & (last_labels != newborn_label)
                if not valid_track.any():
                    continue

                hist_feats = hist_feats[valid_track]
                hist_masks = hist_masks[valid_track]
                last_feats = last_feats[valid_track]
                last_boxes = last_boxes[valid_track]
                last_labels = last_labels[valid_track]

                det_feats = unk_feats[batch_idx, group_idx, frame_idx, det_valid]
                det_boxes = unk_boxes[batch_idx, group_idx, frame_idx, det_valid]
                det_labels = unk_labels[batch_idx, group_idx, frame_idx, det_valid]
                if det_feats.numel() == 0 or hist_feats.numel() == 0:
                    continue

                det_norm = F.normalize(det_feats.to(dtype=work_dtype), dim=-1)
                track_norm = F.normalize(last_feats.to(dtype=work_dtype), dim=-1)
                spatial_scores = torch.matmul(det_norm, track_norm.transpose(0, 1))
                spatial_scores = ((spatial_scores.clamp(min=-1.0, max=1.0) + 1.0) * 0.5).to(dtype=work_dtype)
                motion_scores = _bbox_distance_scores(
                    det_boxes_cxcywh=det_boxes,
                    track_boxes_cxcywh=last_boxes,
                    tau=bbox_tau,
                ).to(dtype=work_dtype)
                det_scores = torch.ones((det_feats.shape[0],), device=device, dtype=work_dtype)

                out = laplace_assoc(
                    spatial_scores=spatial_scores,
                    det_features=det_feats,
                    track_history_features=hist_feats,
                    track_history_masks=hist_masks,
                    motion_scores=motion_scores,
                    det_scores=det_scores,
                )
                fused_scores = out["fused_scores"].to(dtype=work_dtype).clamp(min=1e-4, max=1.0 - 1e-4)
                match_matrix = det_labels.view(-1, 1).eq(last_labels.view(1, -1))

                matched_det = match_matrix.any(dim=1) & (det_labels != newborn_label)
                if matched_det.any():
                    logits = torch.logit(fused_scores[matched_det], eps=1e-4) / temp
                    targets = match_matrix[matched_det].to(dtype=torch.float32).argmax(dim=1)
                    matched_terms.append(F.cross_entropy(logits, targets))
                    matched_rows += int(matched_det.sum().item())

                unmatched_det = ~match_matrix.any(dim=1)
                if unmatched_det.any() and background_weight > 0.0:
                    max_scores = fused_scores[unmatched_det].max(dim=1).values
                    background_terms.append(-torch.log((1.0 - max_scores).clamp(min=1e-4)).mean())
                    background_rows += int(unmatched_det.sum().item())

                active_tracks_total += int(last_labels.numel())
                frames_used += 1

    matched_loss = torch.stack(matched_terms).mean() if matched_terms else zero
    background_loss = torch.stack(background_terms).mean() if background_terms else zero
    loss = matched_loss + float(background_weight) * background_loss

    return {
        "loss": loss,
        "matched_loss": matched_loss,
        "background_loss": background_loss,
        "matched_rows": torch.tensor(float(matched_rows), device=device, dtype=work_dtype),
        "background_rows": torch.tensor(float(background_rows), device=device, dtype=work_dtype),
        "active_tracks": torch.tensor(float(active_tracks_total), device=device, dtype=work_dtype),
        "frames_used": torch.tensor(float(frames_used), device=device, dtype=work_dtype),
    }
