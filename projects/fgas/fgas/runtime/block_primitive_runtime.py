from __future__ import annotations

from dataclasses import dataclass

import torch

from projects.fgas.fgas.model.block_assignment_solver import solve_block_assignment_from_logits
from projects.fgas.fgas.model.block_matcher import BlockMatcherOutput
from projects.fgas.fgas.model.block_primitive import BlockPrimitiveOutput


@dataclass
class BlockPrimitiveDecision:
    takeover: torch.Tensor
    row_assignment: torch.Tensor
    row_no_match: torch.Tensor
    col_newborn: torch.Tensor
    confidence: torch.Tensor


@dataclass
class BlockMatcherDecision:
    takeover: torch.Tensor
    row_assignment: torch.Tensor
    row_no_match: torch.Tensor
    col_newborn: torch.Tensor
    assignment_margin: torch.Tensor
    objective: torch.Tensor


def decode_block_primitive_output(
    output: BlockPrimitiveOutput,
    *,
    edge_mask: torch.Tensor,
    confidence_threshold: float,
) -> BlockPrimitiveDecision:
    edge_logits = output.edge_logits
    row_no_match_logits = output.row_no_match_logits
    col_newborn_logits = output.col_newborn_logits
    confidence = torch.sigmoid(output.block_confidence_logits)
    takeover = confidence >= float(confidence_threshold)
    row_assignment = torch.full(
        (int(edge_logits.shape[0]), int(edge_logits.shape[1])),
        fill_value=-1,
        dtype=torch.long,
        device=edge_logits.device,
    )
    row_no_match = torch.ones(
        (int(edge_logits.shape[0]), int(edge_logits.shape[1])),
        dtype=torch.bool,
        device=edge_logits.device,
    )
    col_newborn = torch.zeros(
        (int(edge_logits.shape[0]), int(edge_logits.shape[2])),
        dtype=torch.bool,
        device=edge_logits.device,
    )
    for batch_idx in range(int(edge_logits.shape[0])):
        solution = solve_block_assignment_from_logits(
            edge_logits=edge_logits[batch_idx],
            row_no_match_logits=row_no_match_logits[batch_idx],
            col_newborn_logits=col_newborn_logits[batch_idx],
            edge_mask=edge_mask[batch_idx],
        )
        row_assignment[batch_idx] = solution.row_assignment
        row_no_match[batch_idx] = solution.row_no_match
        col_newborn[batch_idx] = solution.col_newborn
    return BlockPrimitiveDecision(
        takeover=takeover,
        row_assignment=row_assignment,
        row_no_match=row_no_match,
        col_newborn=col_newborn,
        confidence=confidence,
    )


def decode_block_matcher_output(
    output: BlockMatcherOutput,
    *,
    edge_mask: torch.Tensor,
    margin_threshold: float,
) -> BlockMatcherDecision:
    edge_logits = output.edge_logits
    row_no_match_logits = output.row_no_match_logits
    col_newborn_logits = output.col_newborn_logits
    batch_size = int(edge_logits.shape[0])
    takeover = torch.zeros((batch_size,), dtype=torch.bool, device=edge_logits.device)
    row_assignment = torch.full(
        (batch_size, int(edge_logits.shape[1])),
        fill_value=-1,
        dtype=torch.long,
        device=edge_logits.device,
    )
    row_no_match = torch.ones(
        (batch_size, int(edge_logits.shape[1])),
        dtype=torch.bool,
        device=edge_logits.device,
    )
    col_newborn = torch.zeros(
        (batch_size, int(edge_logits.shape[2])),
        dtype=torch.bool,
        device=edge_logits.device,
    )
    assignment_margin = torch.zeros((batch_size,), dtype=edge_logits.dtype, device=edge_logits.device)
    objective = torch.full((batch_size,), fill_value=float("-inf"), dtype=edge_logits.dtype, device=edge_logits.device)
    for batch_idx in range(batch_size):
        solution = solve_block_assignment_from_logits(
            edge_logits=edge_logits[batch_idx],
            row_no_match_logits=row_no_match_logits[batch_idx],
            col_newborn_logits=col_newborn_logits[batch_idx],
            edge_mask=edge_mask[batch_idx],
        )
        row_assignment[batch_idx] = solution.row_assignment
        row_no_match[batch_idx] = solution.row_no_match
        col_newborn[batch_idx] = solution.col_newborn
        assignment_margin[batch_idx] = float(solution.margin)
        objective[batch_idx] = float(solution.objective)
        takeover[batch_idx] = bool(solution.objective > float("-inf") and solution.margin >= float(margin_threshold))
    return BlockMatcherDecision(
        takeover=takeover,
        row_assignment=row_assignment,
        row_no_match=row_no_match,
        col_newborn=col_newborn,
        assignment_margin=assignment_margin,
        objective=objective,
    )
