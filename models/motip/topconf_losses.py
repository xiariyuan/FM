# Copyright (c) 2024-2026. All Rights Reserved.
# Top Conference Loss Functions for Multi-Object Tracking

import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib
import warnings
from typing import Optional, Tuple, Dict


class TripletLoss(nn.Module):
    """Hard Mining Triplet Loss (from FairMOT/FastReID)"""
    
    def __init__(
        self,
        margin: float = 0.3,
        distance_type: str = 'cosine',
        hard_mining: bool = True,
        normalize: bool = True,
        ignore_label: Optional[int] = None,
    ):
        super().__init__()
        self.margin = margin
        self.distance_type = distance_type
        self.hard_mining = hard_mining
        self.normalize = normalize
        self.ignore_label = int(ignore_label) if ignore_label is not None else None
        
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Flatten if needed
        if embeddings.dim() > 2:
            embeddings = embeddings.reshape(-1, embeddings.shape[-1])
            labels = labels.reshape(-1)
            if masks is not None:
                masks = masks.reshape(-1)
        
        # Filter valid samples
        if masks is not None:
            valid = ~masks
            embeddings = embeddings[valid]
            labels = labels[valid]
        
        # Need at least 2 samples
        if len(embeddings) < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        # Filter invalid IDs (< 0)
        valid_ids = labels >= 0
        if self.ignore_label is not None:
            valid_ids = valid_ids & (labels != int(self.ignore_label))
        if not valid_ids.any():
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        embeddings = embeddings[valid_ids]
        labels = labels[valid_ids]

        if len(embeddings) < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # If there is only one identity in the batch, soft-mining paths can produce NaNs
        # (e.g., softmax over all -inf). Early-exit with a zero loss.
        if int(torch.unique(labels).numel()) < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        # Normalize
        if self.normalize:
            # Use a float16-safe eps; default eps=1e-12 can underflow under autocast and produce NaNs for all-zero rows.
            embeddings = F.normalize(embeddings, p=2, dim=-1, eps=1e-6)
        
        # Compute distance matrix
        if self.distance_type == 'cosine':
            dist_mat = 1.0 - torch.mm(embeddings, embeddings.t())
        else:
            n = embeddings.size(0)
            dist_sq = torch.pow(embeddings, 2).sum(dim=1, keepdim=True).expand(n, n)
            dist_sq = dist_sq + dist_sq.t()
            dist_sq.addmm_(embeddings, embeddings.t(), beta=1, alpha=-2)
            dist_mat = dist_sq.clamp(min=1e-12).sqrt()
        
        n = embeddings.size(0)
        
        # Create positive/negative masks
        labels_eq = labels.unsqueeze(0).eq(labels.unsqueeze(1))
        mask_pos = labels_eq.clone()
        mask_pos.fill_diagonal_(False)
        mask_neg = ~labels_eq
        
        if self.hard_mining:
            dist_ap = dist_mat.masked_fill(~mask_pos, 0).max(dim=1).values
            dist_an = dist_mat.masked_fill(~mask_neg, float('inf')).min(dim=1).values
        else:
            dist_ap_weighted = dist_mat * mask_pos.float()
            weights_ap = F.softmax(dist_ap_weighted, dim=1)
            dist_ap = (dist_ap_weighted * weights_ap).sum(dim=1)
            dist_an_masked = dist_mat.masked_fill(~mask_neg, float('inf'))
            weights_an = F.softmax(-dist_an_masked, dim=1)
            dist_an = (dist_mat * mask_neg.float() * weights_an).sum(dim=1)
        
        # Filter valid triplets
        valid_triplets = (dist_an < float('inf')) & (mask_pos.sum(dim=1) > 0)
        if not valid_triplets.any():
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        dist_ap = dist_ap[valid_triplets]
        dist_an = dist_an[valid_triplets]
        
        # Margin ranking loss
        if self.margin > 0:
            loss = F.relu(dist_ap - dist_an + self.margin).mean()
        else:
            loss = F.softplus(dist_ap - dist_an).mean()
        
        return loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Label Smoothing Cross Entropy Loss"""
    
    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)
        
        with torch.no_grad():
            one_hot = torch.zeros_like(logits)
            one_hot.scatter_(1, targets.unsqueeze(1), 1)
            one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(one_hot * log_probs).sum(dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class BranchConsistencyLoss(nn.Module):
    """Branch Consistency Loss for dual-branch architectures"""
    
    def __init__(self, temperature: float = 1.0, symmetric: bool = True, detach_target: bool = True):
        super().__init__()
        self.temperature = temperature
        self.symmetric = symmetric
        self.detach_target = detach_target
        
    def forward(self, branch1_logits, branch2_logits, masks=None):
        p1 = F.log_softmax(branch1_logits / self.temperature, dim=-1)
        p2 = F.softmax(branch2_logits / self.temperature, dim=-1)
        
        if self.detach_target:
            p2 = p2.detach()
        
        kl_12 = F.kl_div(p1, p2, reduction='none').sum(dim=-1)
        
        if self.symmetric:
            p1_soft = F.softmax(branch1_logits / self.temperature, dim=-1)
            p2_log = F.log_softmax(branch2_logits / self.temperature, dim=-1)
            if self.detach_target:
                p1_soft = p1_soft.detach()
            kl_21 = F.kl_div(p2_log, p1_soft, reduction='none').sum(dim=-1)
            kl = 0.5 * (kl_12 + kl_21)
        else:
            kl = kl_12
        
        if masks is not None:
            kl = kl.masked_fill(masks, 0)
            valid_count = (~masks).float().sum().clamp(min=1)
            return kl.sum() / valid_count
        return kl.mean()


class TPDropFPInsert(nn.Module):
    """
    TP Drop / FP Insert Augmentation (from MeMOTR, ICCV'23)
    
    Randomly drops true positives and inserts false positives during training
    to improve robustness against occlusions and false detections.
    """
    
    def __init__(
        self,
        tp_drop_ratio: float = 0.1,
        fp_insert_ratio: float = 0.3,
    ):
        super().__init__()
        self.tp_drop_ratio = tp_drop_ratio
        self.fp_insert_ratio = fp_insert_ratio
    
    def forward(
        self,
        matched_features: torch.Tensor,
        matched_labels: torch.Tensor,
        unmatched_features: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            matched_features: (N, D) matched detection features
            matched_labels: (N,) corresponding ID labels
            unmatched_features: (M, D) unmatched detection features (for FP insert)
            training: whether in training mode
            
        Returns:
            augmented_features: augmented feature tensor
            augmented_labels: augmented label tensor
        """
        if not training or (self.tp_drop_ratio == 0 and self.fp_insert_ratio == 0):
            return matched_features, matched_labels
        
        device = matched_features.device
        n = len(matched_features)
        
        # TP Drop: randomly drop some true positives
        if self.tp_drop_ratio > 0 and n > 0:
            keep_mask = torch.rand(n, device=device) > self.tp_drop_ratio
            if keep_mask.sum() > 0:
                matched_features = matched_features[keep_mask]
                matched_labels = matched_labels[keep_mask]
        
        # FP Insert: randomly insert false positives
        if self.fp_insert_ratio > 0 and unmatched_features is not None and len(unmatched_features) > 0:
            num_fp = int(len(matched_features) * self.fp_insert_ratio)
            if num_fp > 0 and len(unmatched_features) > 0:
                # Randomly select FPs
                fp_indices = torch.randperm(len(unmatched_features), device=device)[:num_fp]
                fp_features = unmatched_features[fp_indices]
                # FP labels are set to -1 (invalid/newborn)
                fp_labels = torch.full((num_fp,), -1, dtype=matched_labels.dtype, device=device)
                
                matched_features = torch.cat([matched_features, fp_features], dim=0)
                matched_labels = torch.cat([matched_labels, fp_labels], dim=0)
        
        return matched_features, matched_labels


class MemoryBank(nn.Module):
    """
    Long/Short Memory Bank (from MeMOTR, ICCV'23)
    
    Maintains long-term and short-term memory for each track
    to improve temporal consistency.
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        memory_lambda: float = 0.9,
        update_threshold: float = 0.5,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.memory_lambda = memory_lambda
        self.update_threshold = update_threshold
        
        # Confidence weighting network
        self.confidence_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid(),
        )
        
        # Short memory fusion
        self.short_memory_fusion = nn.Sequential(
            nn.Linear(2 * feature_dim, 2 * feature_dim),
            nn.GELU(),
            nn.Linear(2 * feature_dim, feature_dim),
        )
        
        # Memory attention
        self.memory_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            batch_first=True,
        )
        self.memory_norm = nn.LayerNorm(feature_dim)

        # Numerical diagnostics (warn once).
        self._warned_nonfinite_input = False
        self._warned_nonfinite_output = False
        self._warned_large_values = False

    @staticmethod
    def _finite_stats(x: torch.Tensor) -> str:
        x_det = x.detach()
        nonfinite = int((~torch.isfinite(x_det)).sum().item()) if x_det.numel() else 0
        x_num = torch.nan_to_num(x_det.float(), nan=0.0, posinf=0.0, neginf=0.0)
        max_abs = float(x_num.abs().max().item()) if x_num.numel() else 0.0
        x_min = float(x_num.min().item()) if x_num.numel() else 0.0
        x_max = float(x_num.max().item()) if x_num.numel() else 0.0
        return f"shape={tuple(x.shape)} dtype={x.dtype} nonfinite={nonfinite} min={x_min:.3g} max={x_max:.3g} max_abs={max_abs:.3g}"
        
    def update(
        self,
        current_features: torch.Tensor,
        long_memory: torch.Tensor,
        last_features: torch.Tensor,
        scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update memory based on current observations.
        
        Args:
            current_features: (N, D) current frame features
            long_memory: (N, D) long-term memory
            last_features: (N, D) last frame features
            scores: (N,) detection scores
            
        Returns:
            updated_long_memory: updated long-term memory
            query_features: features for next frame query
        """
        # Basic validation (helps catch silent shape bugs early).
        if current_features.dim() != 2:
            raise ValueError(f"[MemoryBank] current_features must be (N,D), got {tuple(current_features.shape)}")
        if long_memory.shape != current_features.shape or last_features.shape != current_features.shape:
            raise ValueError(
                f"[MemoryBank] feature shapes must match: "
                f"current={tuple(current_features.shape)} long={tuple(long_memory.shape)} last={tuple(last_features.shape)}"
            )
        if scores.dim() != 1 or scores.shape[0] != current_features.shape[0]:
            raise ValueError(f"[MemoryBank] scores must be (N,), got {tuple(scores.shape)}")

        # Fail-fast: NaN/Inf in memory-bank inputs will poison the stored state and corrupt all future steps.
        # We stop immediately and surface *which* tensor first went bad.
        bad_inputs = []
        if not torch.isfinite(current_features).all():
            bad_inputs.append(f"current_features({self._finite_stats(current_features)})")
        if not torch.isfinite(long_memory).all():
            bad_inputs.append(f"long_memory({self._finite_stats(long_memory)})")
        if not torch.isfinite(last_features).all():
            bad_inputs.append(f"last_features({self._finite_stats(last_features)})")
        if not torch.isfinite(scores).all():
            bad_inputs.append(f"scores({self._finite_stats(scores)})")
        if bad_inputs:
            raise FloatingPointError(
                "[MemoryBank] Non-finite inputs detected (NaN/Inf). "
                "This indicates an upstream numerical issue. "
                + " | ".join(bad_inputs)
            )

        # IMPORTANT: run the attention/update in float32 with autocast disabled.
        # Memory attention in float16 can overflow/NaN and then permanently pollute the stored memory.
        device_type = "cuda" if current_features.is_cuda else "cpu"
        autocast_off = (
            torch.autocast(device_type=device_type, enabled=False)
            if hasattr(torch, "autocast")
            else contextlib.nullcontext()
        )
        with autocast_off:
            cur = current_features.float()
            long = long_memory.float()
            last = last_features.float()
            scores_f = scores.float()

            # If feature magnitudes are abnormally large (even if finite), attention can overflow and output NaNs.
            # Clamp only in that rare case to keep training stable.
            max_abs_in = float(
                torch.nan_to_num(
                    torch.stack(
                        [
                            cur.detach().abs().max(),
                            long.detach().abs().max(),
                            last.detach().abs().max(),
                        ]
                    ).max(),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).item()
            )
            if max_abs_in > 1e4:
                if not self._warned_large_values:
                    warnings.warn(
                        f"[MemoryBank] Large feature magnitudes detected (max_abs={max_abs_in:.3e}); "
                        "clamping inputs to ±100 to prevent attention NaNs."
                    )
                    self._warned_large_values = True
                cur = cur.clamp(min=-100.0, max=100.0)
                long = long.clamp(min=-100.0, max=100.0)
                last = last.clamp(min=-100.0, max=100.0)

            # Confidence weighting
            confidence_weight = self.confidence_net(cur)
            short_memory = self.short_memory_fusion(
                torch.cat([confidence_weight * cur, last], dim=-1)
            )

            # Memory attention
            is_high_conf = scores_f > float(self.update_threshold)

            if is_high_conf.any():
                # Query from short memory, key/value from long/current memory
                attn_out, _ = self.memory_attn(
                    short_memory[is_high_conf].unsqueeze(0),
                    long[is_high_conf].unsqueeze(0),
                    cur[is_high_conf].unsqueeze(0),
                )
                attn_out = attn_out.squeeze(0)

                query_features = long.clone()
                query_features[is_high_conf] = self.memory_norm(
                    long[is_high_conf] + attn_out
                )
            else:
                query_features = long

            # Update long memory with EMA
            updated_long_memory = long.clone()
            if is_high_conf.any():
                updated_long_memory[is_high_conf] = (
                    (1 - self.memory_lambda) * long[is_high_conf]
                    + self.memory_lambda * cur[is_high_conf]
                )

            # Keep values in a safe range for downstream autocast (fp16) to avoid Inf after casts.
            clamp_val = 100.0
            query_features = query_features.clamp(min=-clamp_val, max=clamp_val)
            updated_long_memory = updated_long_memory.clamp(min=-clamp_val, max=clamp_val)

        if (not torch.isfinite(query_features).all()) or (not torch.isfinite(updated_long_memory).all()):
            # Try to pinpoint which intermediate first went non-finite to speed up debugging.
            debug_parts = [
                f"current_features({self._finite_stats(current_features)})",
                f"long_memory({self._finite_stats(long_memory)})",
                f"last_features({self._finite_stats(last_features)})",
                f"scores({self._finite_stats(scores)})",
                f"confidence_weight({self._finite_stats(confidence_weight)})",
                f"short_memory({self._finite_stats(short_memory)})",
                f"query_features({self._finite_stats(query_features)})",
                f"updated_long_memory({self._finite_stats(updated_long_memory)})",
            ]
            raise FloatingPointError(
                "[MemoryBank] Non-finite outputs detected after update (NaN/Inf). "
                "Stop to avoid contaminating training. "
                + " | ".join(debug_parts)
            )

        return updated_long_memory, query_features

    def forward(
        self,
        current_features: torch.Tensor,
        long_memory: torch.Tensor,
        last_features: torch.Tensor,
        scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward alias of `update()` for clarity/compatibility.

        NOTE: In this codebase MemoryBank is registered as a submodule of the training model before DDP wrapping,
        so calling `update()` directly is still synchronized correctly. Keeping a `forward()` reduces confusion.
        """
        return self.update(
            current_features=current_features,
            long_memory=long_memory,
            last_features=last_features,
            scores=scores,
        )


def build_triplet_loss(config: dict) -> TripletLoss:
    # In this codebase, unknown/newborn label is `NUM_ID_VOCABULARY` (vocab_size = num_id_vocabulary + 1).
    # Triplet should ignore newborn samples; otherwise all newborn detections share the same label and will be
    # mistakenly pulled together as positives.
    ignore_newborn = bool(config.get("TRIPLET_IGNORE_NEWBORN", True))
    ignore_label = config.get("TRIPLET_IGNORE_LABEL", None)
    if ignore_label is None and ignore_newborn and "NUM_ID_VOCABULARY" in config:
        try:
            ignore_label = int(config["NUM_ID_VOCABULARY"])
        except Exception:
            ignore_label = None
    return TripletLoss(
        margin=config.get('TRIPLET_MARGIN', 0.3),
        distance_type=config.get('TRIPLET_DISTANCE', 'cosine'),
        hard_mining=config.get('TRIPLET_HARD_MINING', True),
        normalize=config.get('TRIPLET_NORMALIZE', True),
        ignore_label=ignore_label,
    )


def build_label_smoothing_loss(config: dict) -> LabelSmoothingCrossEntropy:
    return LabelSmoothingCrossEntropy(
        smoothing=config.get('LABEL_SMOOTHING', 0.1),
        reduction='mean',
    )


def build_tp_drop_fp_insert(config: dict) -> TPDropFPInsert:
    return TPDropFPInsert(
        tp_drop_ratio=config.get('TP_DROP_RATIO', 0.1),
        fp_insert_ratio=config.get('FP_INSERT_RATIO', 0.3),
    )


def build_memory_bank(config: dict) -> MemoryBank:
    return MemoryBank(
        feature_dim=config.get('FEATURE_DIM', 256),
        memory_lambda=config.get('MEMORY_LAMBDA', 0.9),
        update_threshold=config.get('MEMORY_UPDATE_THRESHOLD', 0.5),
    )
