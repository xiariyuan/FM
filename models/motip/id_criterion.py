# Copyright (c) Ruopeng Gao. All Rights Reserved.
# Modified to add Label Smoothing support (Top Conference Strategy)

import torch
import einops
import torch.nn as nn
import torch.nn.functional as F

from utils.misc import is_distributed, distributed_world_size
from models.misc import label_to_one_hot


class IDCriterion(nn.Module):
    def __init__(
            self,
            weight: float,
            use_focal_loss: bool,
            label_smoothing: float = 0.0,  # New: Label Smoothing coefficient
    ):
        super().__init__()
        self.weight = weight
        self.use_focal_loss = use_focal_loss
        self.label_smoothing = label_smoothing

        if not self.use_focal_loss:
            # Use PyTorch's built-in label smoothing support
            self.ce_loss = nn.CrossEntropyLoss(
                reduction="none",
                label_smoothing=label_smoothing
            )
        return

    def forward(self, id_logits, id_labels, id_masks):
        if id_labels is None or id_masks is None:
            return torch.tensor(0.0, device=id_logits.device)
        if id_logits.ndim == 4:
            id_logits = id_logits.unsqueeze(2)
        if id_labels.ndim == 3:
            id_labels = id_labels.unsqueeze(2)
        if id_masks.ndim == 3:
            id_masks = id_masks.unsqueeze(2)
        # _B, _G, _T, _N = id_logits.shape
        id_logits = id_logits[:, :, 1:, :, :]
        id_labels = id_labels[:, :, 1:, :]
        id_masks = id_masks[:, :, 1:, :]
        pass

        # Flatten:
        id_logits_flatten = einops.rearrange(id_logits, "b g t n c -> (b g t n) c")
        id_labels_flatten = einops.rearrange(id_labels, "b g t n -> (b g t n)")
        id_masks_flatten = einops.rearrange(id_masks, "b g t n -> (b g t n)")
        # Filter out the invalid id labels:
        id_logits_flatten = id_logits_flatten[~id_masks_flatten]
        id_labels_flatten = id_labels_flatten[~id_masks_flatten]
        # Empty sample protection
        if id_logits_flatten.numel() == 0:
            return torch.tensor(0.0, device=id_logits.device)
        id_labels_flatten = id_labels_flatten.long()
        # Calculate the loss:
        if self.use_focal_loss:
            hard_targets_one_hot = label_to_one_hot(
                id_labels_flatten, n_classes=id_logits_flatten.shape[-1], dtype=id_logits_flatten.dtype
            )
            targets_one_hot = hard_targets_one_hot
            # Apply label smoothing to the CE part if enabled.
            # IMPORTANT: keep focal modulating factor based on hard targets, otherwise smoothing
            # will weaken the (1 - p_t)^gamma term and reduce the benefit of focal loss.
            if self.label_smoothing > 0:
                n_classes = id_logits_flatten.shape[-1]
                targets_one_hot = (
                    targets_one_hot * (1 - self.label_smoothing) +
                    self.label_smoothing / n_classes
                )
            loss = sigmoid_focal_loss(
                inputs=id_logits_flatten,
                targets=targets_one_hot,
                pt_targets=hard_targets_one_hot,
            ).sum()
        else:
            loss = self.ce_loss(id_logits_flatten, id_labels_flatten).sum()
        num_ids = torch.as_tensor([len(id_logits_flatten)], dtype=torch.float, device=id_logits.device)

        if is_distributed():
            torch.distributed.all_reduce(num_ids)
        num_ids = torch.clamp(num_ids / distributed_world_size(), min=1).item()

        return loss / num_ids


def sigmoid_focal_loss(inputs, targets, pt_targets=None, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    pt = pt_targets if pt_targets is not None else targets
    p_t = prob * pt + (1 - prob) * (1 - pt)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * pt + (1 - alpha) * (1 - pt)
        loss = alpha_t * loss

    return loss.mean(1).sum()


def build(config: dict):
    return IDCriterion(
        weight=config["ID_LOSS_WEIGHT"],
        use_focal_loss=config["USE_FOCAL_LOSS"],
        label_smoothing=config.get("LABEL_SMOOTHING", 0.0),  # New parameter
    )
