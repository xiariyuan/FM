from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch


def binary_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    positives = [(float(score), int(label)) for score, label in zip(scores, labels) if int(label) == 1]
    negatives = [(float(score), int(label)) for score, label in zip(scores, labels) if int(label) == 0]
    if not positives or not negatives:
        return 0.0
    ordered = sorted(zip(scores, labels), key=lambda item: float(item[0]))
    rank_sum = 0.0
    positive_count = 0
    for idx, (_, label) in enumerate(ordered, start=1):
        if int(label) == 1:
            rank_sum += float(idx)
            positive_count += 1
    negative_count = len(ordered) - positive_count
    if positive_count == 0 or negative_count == 0:
        return 0.0
    auc = (rank_sum - positive_count * (positive_count + 1) / 2.0) / (positive_count * negative_count)
    return float(max(0.0, min(1.0, auc)))


def group_top1_accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, ambiguous_only: bool = False, ambiguous: torch.Tensor | None = None) -> float:
    valid_groups = 0
    correct = 0
    for idx in range(logits.shape[0]):
        if ambiguous_only and ambiguous is not None and not bool(ambiguous[idx].item()):
            continue
        valid_mask = mask[idx]
        if not bool(valid_mask.any().item()):
            continue
        valid_groups += 1
        row_logits = logits[idx].masked_fill(~valid_mask, float("-inf"))
        pred_idx = int(torch.argmax(row_logits).item())
        if int(labels[idx, pred_idx].item()) > 0:
            correct += 1
    return float(correct / max(valid_groups, 1))


def edit_flip_stats(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, baseline_scores: torch.Tensor) -> Dict[str, float]:
    wrong_to_right = 0
    right_to_wrong = 0
    total = 0
    for idx in range(logits.shape[0]):
        valid_mask = mask[idx]
        if not bool(valid_mask.any().item()):
            continue
        total += 1
        base_row = baseline_scores[idx].masked_fill(~valid_mask, float("-inf"))
        pred_row = logits[idx].masked_fill(~valid_mask, float("-inf"))
        base_idx = int(torch.argmax(base_row).item())
        pred_idx = int(torch.argmax(pred_row).item())
        base_ok = int(labels[idx, base_idx].item()) > 0
        pred_ok = int(labels[idx, pred_idx].item()) > 0
        if (not base_ok) and pred_ok:
            wrong_to_right += 1
        if base_ok and (not pred_ok):
            right_to_wrong += 1
    return {
        "wrong_to_right_rate": float(wrong_to_right / max(total, 1)),
        "right_to_wrong_rate": float(right_to_wrong / max(total, 1)),
    }


def flatten_scores(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> Dict[str, List[float]]:
    flat_scores: List[float] = []
    flat_labels: List[int] = []
    for idx in range(logits.shape[0]):
        valid_mask = mask[idx]
        for cand_idx in torch.nonzero(valid_mask, as_tuple=False).view(-1).tolist():
            flat_scores.append(float(logits[idx, cand_idx].item()))
            flat_labels.append(int(labels[idx, cand_idx].item()))
    return {"scores": flat_scores, "labels": flat_labels}
