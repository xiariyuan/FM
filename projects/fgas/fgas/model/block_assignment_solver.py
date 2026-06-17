from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Set, Tuple

import torch
import torch.nn.functional as F


@dataclass
class BlockAssignmentSolution:
    row_assignment: torch.Tensor
    row_no_match: torch.Tensor
    col_newborn: torch.Tensor
    objective: float
    margin: float


AssignmentSignature = Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]


def assignment_signature(
    *,
    row_assignment: torch.Tensor,
    row_no_match: torch.Tensor,
    col_newborn: torch.Tensor,
) -> AssignmentSignature:
    row_assignment_cpu = row_assignment.detach().to(device="cpu", dtype=torch.long)
    row_no_match_cpu = row_no_match.detach().to(device="cpu", dtype=torch.bool)
    col_newborn_cpu = col_newborn.detach().to(device="cpu", dtype=torch.bool)
    return (
        tuple(int(v) for v in row_assignment_cpu.tolist()),
        tuple(int(bool(v)) for v in row_no_match_cpu.tolist()),
        tuple(int(bool(v)) for v in col_newborn_cpu.tolist()),
    )


def score_assignment_from_logits(
    *,
    edge_logits: torch.Tensor,
    row_no_match_logits: torch.Tensor,
    col_newborn_logits: torch.Tensor,
    row_assignment: torch.Tensor,
    row_no_match: torch.Tensor,
    col_newborn: torch.Tensor,
    edge_mask: torch.Tensor,
) -> torch.Tensor:
    row_count = int(edge_logits.shape[0])
    col_count = int(edge_logits.shape[1])
    total = edge_logits.new_zeros(())
    for row_idx in range(row_count):
        if not bool(edge_mask[row_idx].any()):
            continue
        if bool(row_no_match[row_idx]):
            total = total + F.logsigmoid(row_no_match_logits[row_idx])
            continue
        col_idx = int(row_assignment[row_idx].item())
        if col_idx < 0 or col_idx >= col_count or not bool(edge_mask[row_idx, col_idx]):
            continue
        total = total + F.logsigmoid(-row_no_match_logits[row_idx]) + F.logsigmoid(edge_logits[row_idx, col_idx])
    for col_idx in range(col_count):
        if not bool(edge_mask[:, col_idx].any()):
            continue
        if bool(col_newborn[col_idx]):
            total = total + F.logsigmoid(col_newborn_logits[col_idx])
        else:
            total = total + F.logsigmoid(-col_newborn_logits[col_idx])
    return total


def solve_block_assignment_from_logits(
    *,
    edge_logits: torch.Tensor,
    row_no_match_logits: torch.Tensor,
    col_newborn_logits: torch.Tensor,
    edge_mask: torch.Tensor,
    forbidden_signature: Optional[AssignmentSignature] = None,
) -> BlockAssignmentSolution:
    device = edge_logits.device
    row_count = int(edge_logits.shape[0])
    col_count = int(edge_logits.shape[1])

    valid_mask = edge_mask.detach().to(dtype=torch.bool, device="cpu")
    edge_pos = F.logsigmoid(edge_logits.detach().to(device="cpu")).to(dtype=torch.float32)
    row_match_score = F.logsigmoid((-row_no_match_logits).detach().to(device="cpu")).to(dtype=torch.float32)
    row_nomatch_score = F.logsigmoid(row_no_match_logits.detach().to(device="cpu")).to(dtype=torch.float32)
    col_matched_score = F.logsigmoid((-col_newborn_logits).detach().to(device="cpu")).to(dtype=torch.float32)
    col_newborn_score = F.logsigmoid(col_newborn_logits.detach().to(device="cpu")).to(dtype=torch.float32)

    present_rows = [row_idx for row_idx in range(row_count) if bool(valid_mask[row_idx].any().item())]
    present_cols = [col_idx for col_idx in range(col_count) if bool(valid_mask[:, col_idx].any().item())]

    best_score = float("-inf")
    second_score = float("-inf")
    best_signature: Optional[AssignmentSignature] = None
    best_assignment = [-1 for _ in range(row_count)]
    current_assignment = [-1 for _ in range(row_count)]

    def finalize(used_cols: Set[int], partial_score: float) -> None:
        nonlocal best_score, second_score, best_assignment, best_signature
        total_score = float(partial_score)
        for col_idx in present_cols:
            if int(col_idx) in used_cols:
                total_score += float(col_matched_score[col_idx].item())
            else:
                total_score += float(col_newborn_score[col_idx].item())
        row_assignment_tuple = tuple(int(v) for v in current_assignment)
        row_no_match_tuple = tuple(int(v < 0) for v in current_assignment)
        col_newborn_tuple = tuple(int(col_idx not in used_cols) if col_idx in present_cols else 0 for col_idx in range(col_count))
        current_signature: AssignmentSignature = (
            row_assignment_tuple,
            row_no_match_tuple,
            col_newborn_tuple,
        )
        if forbidden_signature is not None and current_signature == forbidden_signature:
            return
        if total_score > best_score:
            second_score = best_score
            best_score = total_score
            best_assignment = list(current_assignment)
            best_signature = current_signature
        elif total_score > second_score:
            second_score = total_score

    def dfs(depth: int, used_cols: Set[int], partial_score: float) -> None:
        if depth >= len(present_rows):
            finalize(used_cols, partial_score)
            return
        row_idx = int(present_rows[depth])

        current_assignment[row_idx] = -1
        dfs(
            depth + 1,
            used_cols,
            float(partial_score) + float(row_nomatch_score[row_idx].item()),
        )

        valid_cols = [col_idx for col_idx in range(col_count) if bool(valid_mask[row_idx, col_idx].item())]
        valid_cols.sort(key=lambda col_idx: float(edge_pos[row_idx, col_idx].item()), reverse=True)
        for col_idx in valid_cols:
            if int(col_idx) in used_cols:
                continue
            current_assignment[row_idx] = int(col_idx)
            dfs(
                depth + 1,
                used_cols | {int(col_idx)},
                float(partial_score)
                + float(row_match_score[row_idx].item())
                + float(edge_pos[row_idx, col_idx].item()),
            )

    if present_rows:
        dfs(depth=0, used_cols=set(), partial_score=0.0)
    else:
        finalize(set(), 0.0)

    row_assignment = torch.full((row_count,), fill_value=-1, dtype=torch.long)
    for row_idx, col_idx in enumerate(best_assignment):
        row_assignment[row_idx] = int(col_idx)
    row_no_match = row_assignment < 0
    col_newborn = torch.zeros((col_count,), dtype=torch.bool)
    used_cols = {int(col_idx) for col_idx in best_assignment if int(col_idx) >= 0}
    for col_idx in present_cols:
        col_newborn[col_idx] = int(col_idx) not in used_cols

    if best_signature is None:
        row_assignment = torch.full((row_count,), fill_value=-1, dtype=torch.long)
        row_no_match = torch.ones((row_count,), dtype=torch.bool)
        col_newborn = torch.zeros((col_count,), dtype=torch.bool)
        for col_idx in present_cols:
            col_newborn[col_idx] = True
        return BlockAssignmentSolution(
            row_assignment=row_assignment.to(device=device),
            row_no_match=row_no_match.to(device=device),
            col_newborn=col_newborn.to(device=device),
            objective=float("-inf"),
            margin=0.0,
        )

    margin = float(best_score - second_score) if second_score > float("-inf") else float("inf")
    return BlockAssignmentSolution(
        row_assignment=row_assignment.to(device=device),
        row_no_match=row_no_match.to(device=device),
        col_newborn=col_newborn.to(device=device),
        objective=float(best_score),
        margin=float(margin),
    )
