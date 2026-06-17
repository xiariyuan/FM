#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.fgas.fgas.data.blockbank_dataset import ConflictBlockDataset, collate_blockbank_batch
from projects.fgas.fgas.model.block_assignment_solver import (
    assignment_signature,
    score_assignment_from_logits,
    solve_block_assignment_from_logits,
)
from projects.fgas.fgas.model.block_matcher import FGASTrueBlockMatcher, MODEL_FAMILY, default_num_stages
from projects.fgas.fgas.model.block_resolver import STAGE_NAME_TO_ID


REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"
SUMMARY_FIELDS = [
    "train_jsonl",
    "val_jsonl",
    "init_checkpoint",
    "init_loaded_tensor_count",
    "feature_mode",
    "model_variant",
    "edge_logit_mode",
    "base_logit_scale",
    "side_init_scale",
    "edge_aux_feature_names",
    "row_aux_feature_names",
    "col_aux_feature_names",
    "input_dim",
    "epochs",
    "batch_size",
    "lr",
    "weight_decay",
    "hidden_dim",
    "stage_embed_dim",
    "num_heads",
    "num_layers",
    "dropout",
    "ambiguous_oversample",
    "block_margin",
    "block_margin_weight",
    "edge_bce_weight",
    "row_ce_weight",
    "col_bce_weight",
    "row_context_dim",
    "col_context_dim",
    "best_epoch",
    "best_metric",
    "val_exact_block_acc",
    "val_ambiguous_exact_block_acc",
    "val_row_top1",
    "val_ambiguous_row_top1",
    "val_structured_loss",
    "val_edge_bce",
    "val_row_ce",
    "val_col_bce",
    "status",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a true block-level FGAS matcher with structured assignment supervision.")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--feature-mode", choices=["full", "nofreq"], default="nofreq")
    parser.add_argument("--model-variant", choices=["standard", "hybrid_v3core_v4aux"], default="standard")
    parser.add_argument("--edge-logit-mode", choices=["direct", "base_residual"], default="direct")
    parser.add_argument("--base-logit-scale", type=float, default=0.35)
    parser.add_argument("--side-init-scale", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--stage-embed-dim", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--block-margin", type=float, default=0.2)
    parser.add_argument("--block-margin-weight", type=float, default=1.0)
    parser.add_argument("--edge-bce-weight", type=float, default=0.25)
    parser.add_argument("--row-ce-weight", type=float, default=0.5)
    parser.add_argument("--col-bce-weight", type=float, default=0.1)
    parser.add_argument("--ambiguous-oversample", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    return parser.parse_args()


def write_single_row_csv(path: Path, fieldnames: List[str], row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def append_registry(args: argparse.Namespace, summary_csv: Path, checkpoint: Path | None, status: str, notes: str) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/append_experiment_record.py"),
        "--csv",
        str(args.registry_csv),
        "--kind",
        "train",
        "--status",
        status,
        "--script",
        "scripts/train_fgas_block_matcher.py",
        "--dataset",
        "MOT17",
        "--split",
        "blockbank",
        "--tracker-family",
        "deep_ocsort_fgas",
        "--variant",
        Path(args.out_dir).name,
        "--tag",
        Path(args.out_dir).name,
        "--run-root",
        str(Path(args.out_dir)),
        "--summary-csv",
        str(summary_csv),
        "--checkpoint",
        str(checkpoint) if checkpoint else "",
        "--notes",
        notes,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def load_matching_init_checkpoint(model: nn.Module, checkpoint_path: str) -> int:
    payload = torch.load(str(checkpoint_path), map_location="cpu")
    model_state = payload.get("model_state", payload)
    current_state = model.state_dict()
    matched = {
        key: value
        for key, value in model_state.items()
        if key in current_state and tuple(current_state[key].shape) == tuple(value.shape)
    }
    model.load_state_dict(matched, strict=False)
    return int(len(matched))


def set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def stage_ids_from_names(stage_names: List[str], device: torch.device) -> torch.Tensor:
    ids = [int(STAGE_NAME_TO_ID.get(str(name), STAGE_NAME_TO_ID["primary"])) for name in stage_names]
    return torch.tensor(ids, dtype=torch.long, device=device)


def select_feature_names(all_feature_names: List[str], feature_mode: str) -> List[str]:
    if str(feature_mode) == "nofreq":
        return [str(name) for name in all_feature_names if str(name) not in {"s_low", "s_mid", "s_high"}]
    return [str(name) for name in all_feature_names]


def select_feature_indices(all_feature_names: List[str], feature_mode: str) -> List[int]:
    kept = select_feature_names(all_feature_names, feature_mode)
    return [int(all_feature_names.index(name)) for name in kept]


def hybrid_aux_feature_splits(
    kept_edge_feature_names: List[str],
    row_feature_names: List[str],
    col_feature_names: List[str],
) -> tuple[List[str], List[str], List[str]]:
    edge_aux = [
        name
        for name in ["track_age_short_norm", "track_fresh_flag", "track_stale_flag"]
        if name in kept_edge_feature_names
    ]
    row_aux = [
        name
        for name in ["row_track_age_short_norm", "row_track_fresh_flag", "row_track_stale_flag"]
        if name in row_feature_names
    ]
    col_aux = [
        name
        for name in [
            "col_best_track_age_short_norm",
            "col_best_track_fresh_flag",
            "col_best_track_stale_flag",
            "col_second_track_age_short_norm",
            "col_second_track_fresh_flag",
            "col_second_track_stale_flag",
        ]
        if name in col_feature_names
    ]
    return edge_aux, row_aux, col_aux


def row_ce_loss(
    edge_logits: torch.Tensor,
    row_no_match_logits: torch.Tensor,
    row_targets: torch.Tensor,
    row_mask: torch.Tensor,
) -> torch.Tensor:
    batch_size, row_count, col_count = edge_logits.shape
    no_match_index = col_count
    target = row_targets.clone()
    target[target < 0] = no_match_index
    logits = torch.cat([edge_logits, row_no_match_logits.unsqueeze(-1)], dim=-1)
    flat_logits = logits.view(batch_size * row_count, col_count + 1)
    flat_targets = target.view(batch_size * row_count)
    flat_mask = row_mask.view(batch_size * row_count)
    if not bool(flat_mask.any()):
        return flat_logits.sum() * 0.0
    return nn.functional.cross_entropy(flat_logits[flat_mask], flat_targets[flat_mask])


def edge_bce_loss(edge_logits: torch.Tensor, edge_labels: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
    if not bool(edge_mask.any()):
        return edge_logits.sum() * 0.0
    return nn.functional.binary_cross_entropy_with_logits(edge_logits[edge_mask], edge_labels[edge_mask])


def col_bce_loss(col_logits: torch.Tensor, col_targets: torch.Tensor, col_mask: torch.Tensor) -> torch.Tensor:
    if not bool(col_mask.any()):
        return col_logits.sum() * 0.0
    return nn.functional.binary_cross_entropy_with_logits(col_logits[col_mask], col_targets[col_mask])


def gt_assignment_tensors(
    *,
    row_targets: torch.Tensor,
    col_targets: torch.Tensor,
    row_mask: torch.Tensor,
    col_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    row_count = int(row_mask.sum().item())
    col_count = int(col_mask.sum().item())
    row_assignment = torch.full((row_count,), fill_value=-1, dtype=torch.long, device=row_targets.device)
    row_no_match = torch.ones((row_count,), dtype=torch.bool, device=row_targets.device)
    for row_idx in range(row_count):
        target = int(row_targets[row_idx].item())
        if target >= 0:
            row_assignment[row_idx] = int(target)
            row_no_match[row_idx] = False
    col_newborn = (col_targets[:col_count] >= 0.5).to(dtype=torch.bool)
    return row_assignment, row_no_match, col_newborn


def structured_block_loss(
    *,
    edge_logits: torch.Tensor,
    row_no_match_logits: torch.Tensor,
    col_newborn_logits: torch.Tensor,
    edge_mask: torch.Tensor,
    row_targets: torch.Tensor,
    col_targets: torch.Tensor,
    row_mask: torch.Tensor,
    col_mask: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    row_count = int(row_mask.sum().item())
    col_count = int(col_mask.sum().item())
    if row_count <= 0 or col_count <= 0:
        return edge_logits.new_zeros(())
    gt_row_assignment, gt_row_no_match, gt_col_newborn = gt_assignment_tensors(
        row_targets=row_targets[:row_count],
        col_targets=col_targets[:col_count],
        row_mask=row_mask[:row_count],
        col_mask=col_mask[:col_count],
    )
    gt_score = score_assignment_from_logits(
        edge_logits=edge_logits[:row_count, :col_count],
        row_no_match_logits=row_no_match_logits[:row_count],
        col_newborn_logits=col_newborn_logits[:col_count],
        row_assignment=gt_row_assignment,
        row_no_match=gt_row_no_match,
        col_newborn=gt_col_newborn,
        edge_mask=edge_mask[:row_count, :col_count],
    )
    competitor = solve_block_assignment_from_logits(
        edge_logits=edge_logits[:row_count, :col_count],
        row_no_match_logits=row_no_match_logits[:row_count],
        col_newborn_logits=col_newborn_logits[:col_count],
        edge_mask=edge_mask[:row_count, :col_count],
        forbidden_signature=assignment_signature(
            row_assignment=gt_row_assignment,
            row_no_match=gt_row_no_match,
            col_newborn=gt_col_newborn,
        ),
    )
    if competitor.objective == float("-inf"):
        return gt_score.new_zeros(())
    competitor_score = score_assignment_from_logits(
        edge_logits=edge_logits[:row_count, :col_count],
        row_no_match_logits=row_no_match_logits[:row_count],
        col_newborn_logits=col_newborn_logits[:col_count],
        row_assignment=competitor.row_assignment,
        row_no_match=competitor.row_no_match,
        col_newborn=competitor.col_newborn,
        edge_mask=edge_mask[:row_count, :col_count],
    )
    return torch.relu(torch.tensor(float(margin), dtype=gt_score.dtype, device=gt_score.device) - gt_score + competitor_score)


def exact_match(
    *,
    pred_row_assignment: torch.Tensor,
    pred_row_no_match: torch.Tensor,
    pred_col_newborn: torch.Tensor,
    row_targets: torch.Tensor,
    col_targets: torch.Tensor,
    row_mask: torch.Tensor,
    col_mask: torch.Tensor,
) -> bool:
    gt_row_assignment, gt_row_no_match, gt_col_newborn = gt_assignment_tensors(
        row_targets=row_targets,
        col_targets=col_targets,
        row_mask=row_mask,
        col_mask=col_mask,
    )
    return bool(
        torch.equal(pred_row_assignment[: gt_row_assignment.shape[0]].cpu(), gt_row_assignment.cpu())
        and torch.equal(pred_row_no_match[: gt_row_no_match.shape[0]].cpu(), gt_row_no_match.cpu())
        and torch.equal(pred_col_newborn[: gt_col_newborn.shape[0]].cpu(), gt_col_newborn.cpu())
    )


def evaluate(
    *,
    model: FGASTrueBlockMatcher,
    loader: DataLoader,
    device: torch.device,
    feature_indices: torch.Tensor,
    margin: float,
) -> Dict[str, float]:
    model.eval()
    total_blocks = 0
    correct_blocks = 0
    ambiguous_total_blocks = 0
    ambiguous_correct_blocks = 0
    total_rows = 0
    correct_rows = 0
    ambiguous_total_rows = 0
    ambiguous_correct_rows = 0
    total_structured = 0.0
    total_edge_bce = 0.0
    total_row_ce = 0.0
    total_col_bce = 0.0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            edge_features_full = batch["edge_features"].to(device)
            edge_features = edge_features_full.index_select(dim=-1, index=feature_indices)
            edge_mask = batch["edge_mask"].to(device)
            row_mask = batch["row_mask"].to(device)
            row_targets = batch["row_targets"].to(device)
            col_mask = batch["col_mask"].to(device)
            col_targets = batch["col_newborn_targets"].to(device)
            edge_labels = batch["edge_labels"].to(device)
            ambiguous = batch["ambiguous"].to(device)
            stage_ids = stage_ids_from_names(list(batch["stage_names"]), device)
            output = model(
                edge_features=edge_features,
                edge_mask=edge_mask,
                stage_ids=stage_ids,
                row_context=batch["row_features"].to(device),
                col_context=batch["col_features"].to(device),
            )
            batch_structured = output.edge_logits.new_zeros(())
            for batch_idx in range(int(edge_features.shape[0])):
                batch_structured = batch_structured + structured_block_loss(
                    edge_logits=output.edge_logits[batch_idx],
                    row_no_match_logits=output.row_no_match_logits[batch_idx],
                    col_newborn_logits=output.col_newborn_logits[batch_idx],
                    edge_mask=edge_mask[batch_idx],
                    row_targets=row_targets[batch_idx],
                    col_targets=col_targets[batch_idx],
                    row_mask=row_mask[batch_idx],
                    col_mask=col_mask[batch_idx],
                    margin=float(margin),
                )
                row_count = int(row_mask[batch_idx].sum().item())
                col_count = int(col_mask[batch_idx].sum().item())
                if row_count <= 0 or col_count <= 0:
                    continue
                pred = solve_block_assignment_from_logits(
                    edge_logits=output.edge_logits[batch_idx, :row_count, :col_count],
                    row_no_match_logits=output.row_no_match_logits[batch_idx, :row_count],
                    col_newborn_logits=output.col_newborn_logits[batch_idx, :col_count],
                    edge_mask=edge_mask[batch_idx, :row_count, :col_count],
                )
                is_exact = exact_match(
                    pred_row_assignment=pred.row_assignment,
                    pred_row_no_match=pred.row_no_match,
                    pred_col_newborn=pred.col_newborn,
                    row_targets=row_targets[batch_idx, :row_count],
                    col_targets=col_targets[batch_idx, :col_count],
                    row_mask=row_mask[batch_idx, :row_count],
                    col_mask=col_mask[batch_idx, :col_count],
                )
                total_blocks += 1
                correct_blocks += int(is_exact)
                if bool(ambiguous[batch_idx].item()):
                    ambiguous_total_blocks += 1
                    ambiguous_correct_blocks += int(is_exact)

                no_match_index = col_count
                pred_rows = pred.row_assignment.clone()
                pred_rows[pred.row_no_match] = int(no_match_index)
                gt_rows = row_targets[batch_idx, :row_count].clone()
                gt_rows[gt_rows < 0] = int(no_match_index)
                row_correct = pred_rows == gt_rows
                correct_rows += int(row_correct.sum().item())
                total_rows += int(row_count)
                if bool(ambiguous[batch_idx].item()):
                    ambiguous_correct_rows += int(row_correct.sum().item())
                    ambiguous_total_rows += int(row_count)

            edge_loss = edge_bce_loss(output.edge_logits, edge_labels, edge_mask)
            row_loss = row_ce_loss(output.edge_logits, output.row_no_match_logits, row_targets, row_mask)
            col_loss = col_bce_loss(output.col_newborn_logits, col_targets, col_mask)
            total_structured += float(batch_structured.item() / max(int(edge_features.shape[0]), 1))
            total_edge_bce += float(edge_loss.item())
            total_row_ce += float(row_loss.item())
            total_col_bce += float(col_loss.item())
            batches += 1

    return {
        "exact_block_acc": float(correct_blocks / total_blocks) if total_blocks else 0.0,
        "ambiguous_exact_block_acc": float(ambiguous_correct_blocks / ambiguous_total_blocks) if ambiguous_total_blocks else 0.0,
        "row_top1": float(correct_rows / total_rows) if total_rows else 0.0,
        "ambiguous_row_top1": float(ambiguous_correct_rows / ambiguous_total_rows) if ambiguous_total_rows else 0.0,
        "structured_loss": float(total_structured / batches) if batches else 0.0,
        "edge_bce": float(total_edge_bce / batches) if batches else 0.0,
        "row_ce": float(total_row_ce / batches) if batches else 0.0,
        "col_bce": float(total_col_bce / batches) if batches else 0.0,
    }


def main() -> int:
    args = parse_args()
    set_seed(int(args.seed))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary.csv"
    metrics_jsonl = out_dir / "metrics.jsonl"
    best_ckpt = out_dir / "best.pt"
    summary_row: Dict[str, object] = {
        "train_jsonl": str(Path(args.train_jsonl)),
        "val_jsonl": str(Path(args.val_jsonl)),
        "init_checkpoint": str(args.init_checkpoint or ""),
        "init_loaded_tensor_count": 0,
        "feature_mode": str(args.feature_mode),
        "model_variant": str(args.model_variant),
        "edge_logit_mode": str(args.edge_logit_mode),
        "base_logit_scale": float(args.base_logit_scale),
        "side_init_scale": float(args.side_init_scale),
        "edge_aux_feature_names": "",
        "row_aux_feature_names": "",
        "col_aux_feature_names": "",
        "input_dim": 0,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "hidden_dim": int(args.hidden_dim),
        "stage_embed_dim": int(args.stage_embed_dim),
        "num_heads": int(args.num_heads),
        "num_layers": int(args.num_layers),
        "dropout": float(args.dropout),
        "ambiguous_oversample": float(args.ambiguous_oversample),
        "block_margin": float(args.block_margin),
        "block_margin_weight": float(args.block_margin_weight),
        "edge_bce_weight": float(args.edge_bce_weight),
        "row_ce_weight": float(args.row_ce_weight),
        "col_bce_weight": float(args.col_bce_weight),
        "row_context_dim": 0,
        "col_context_dim": 0,
        "best_epoch": -1,
        "best_metric": 0.0,
        "val_exact_block_acc": 0.0,
        "val_ambiguous_exact_block_acc": 0.0,
        "val_row_top1": 0.0,
        "val_ambiguous_row_top1": 0.0,
        "val_structured_loss": float("inf"),
        "val_edge_bce": float("inf"),
        "val_row_ce": float("inf"),
        "val_col_bce": float("inf"),
        "status": "running",
        "error": "",
    }
    write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
    append_registry(args, summary_csv, best_ckpt, "running", "FGAS true block matcher training")

    try:
        train_dataset = ConflictBlockDataset.from_jsonl(
            args.train_jsonl,
            ambiguous_oversample=float(args.ambiguous_oversample),
            seed=int(args.seed),
        )
        val_dataset = ConflictBlockDataset.from_jsonl(args.val_jsonl, ambiguous_oversample=1.0, seed=int(args.seed))
        if len(train_dataset.blocks) == 0 or len(val_dataset.blocks) == 0:
            raise ValueError("Empty block-bank dataset.")

        first_block = train_dataset.blocks[0]
        all_feature_names = list(first_block.edges[0].feature_names)
        kept_feature_names = select_feature_names(all_feature_names, str(args.feature_mode))
        feature_indices_list = select_feature_indices(all_feature_names, str(args.feature_mode))
        feature_indices = torch.tensor(feature_indices_list, dtype=torch.long, device=str(args.device))

        train_loader = DataLoader(
            train_dataset,
            batch_size=int(args.batch_size),
            shuffle=True,
            num_workers=0,
            collate_fn=collate_blockbank_batch,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=0,
            collate_fn=collate_blockbank_batch,
        )

        device = torch.device(str(args.device))
        input_dim = len(feature_indices_list)
        use_base_residual = str(args.edge_logit_mode) == "base_residual"
        base_score_index = int(kept_feature_names.index("base_similarity")) if use_base_residual else -1
        edge_aux_feature_names: List[str] = []
        row_aux_feature_names: List[str] = []
        col_aux_feature_names: List[str] = []
        if str(args.model_variant) == "hybrid_v3core_v4aux":
            edge_aux_feature_names, row_aux_feature_names, col_aux_feature_names = hybrid_aux_feature_splits(
                kept_edge_feature_names=kept_feature_names,
                row_feature_names=list(first_block.row_feature_names),
                col_feature_names=list(first_block.col_feature_names),
            )
            if len(edge_aux_feature_names) <= 0 or len(row_aux_feature_names) <= 0 or len(col_aux_feature_names) <= 0:
                raise ValueError("hybrid_v3core_v4aux requires v4-style auxiliary edge/row/col features")
        summary_row["input_dim"] = int(input_dim)
        summary_row["row_context_dim"] = int(len(first_block.row_feature_names))
        summary_row["col_context_dim"] = int(len(first_block.col_feature_names))
        summary_row["edge_aux_feature_names"] = "|".join(edge_aux_feature_names)
        summary_row["row_aux_feature_names"] = "|".join(row_aux_feature_names)
        summary_row["col_aux_feature_names"] = "|".join(col_aux_feature_names)
        write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)

        model = FGASTrueBlockMatcher(
            input_dim=int(input_dim),
            row_context_dim=int(len(first_block.row_feature_names)),
            col_context_dim=int(len(first_block.col_feature_names)),
            hidden_dim=int(args.hidden_dim),
            stage_embed_dim=int(args.stage_embed_dim),
            num_stages=default_num_stages(),
            num_heads=int(args.num_heads),
            num_layers=int(args.num_layers),
            dropout=float(args.dropout),
            use_base_residual=use_base_residual,
            base_score_index=base_score_index,
            base_logit_scale=float(args.base_logit_scale),
            edge_aux_indices=[int(kept_feature_names.index(name)) for name in edge_aux_feature_names],
            row_aux_indices=[int(first_block.row_feature_names.index(name)) for name in row_aux_feature_names],
            col_aux_indices=[int(first_block.col_feature_names.index(name)) for name in col_aux_feature_names],
            side_init_scale=float(args.side_init_scale),
        ).to(device)
        if str(args.init_checkpoint or "").strip():
            summary_row["init_loaded_tensor_count"] = load_matching_init_checkpoint(model, str(args.init_checkpoint))
            write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

        best_metric = -1.0
        with metrics_jsonl.open("w", encoding="utf-8") as handle:
            for epoch in range(1, int(args.epochs) + 1):
                model.train()
                for batch in train_loader:
                    edge_features_full = batch["edge_features"].to(device)
                    edge_features = edge_features_full.index_select(dim=-1, index=feature_indices)
                    edge_mask = batch["edge_mask"].to(device)
                    row_mask = batch["row_mask"].to(device)
                    row_targets = batch["row_targets"].to(device)
                    col_mask = batch["col_mask"].to(device)
                    col_targets = batch["col_newborn_targets"].to(device)
                    edge_labels = batch["edge_labels"].to(device)
                    stage_ids = stage_ids_from_names(list(batch["stage_names"]), device)
                    output = model(
                        edge_features=edge_features,
                        edge_mask=edge_mask,
                        stage_ids=stage_ids,
                        row_context=batch["row_features"].to(device),
                        col_context=batch["col_features"].to(device),
                    )
                    structured_loss = output.edge_logits.new_zeros(())
                    for batch_idx in range(int(edge_features.shape[0])):
                        structured_loss = structured_loss + structured_block_loss(
                            edge_logits=output.edge_logits[batch_idx],
                            row_no_match_logits=output.row_no_match_logits[batch_idx],
                            col_newborn_logits=output.col_newborn_logits[batch_idx],
                            edge_mask=edge_mask[batch_idx],
                            row_targets=row_targets[batch_idx],
                            col_targets=col_targets[batch_idx],
                            row_mask=row_mask[batch_idx],
                            col_mask=col_mask[batch_idx],
                            margin=float(args.block_margin),
                        )
                    structured_loss = structured_loss / max(int(edge_features.shape[0]), 1)
                    edge_loss = edge_bce_loss(output.edge_logits, edge_labels, edge_mask)
                    row_loss = row_ce_loss(output.edge_logits, output.row_no_match_logits, row_targets, row_mask)
                    col_loss = col_bce_loss(output.col_newborn_logits, col_targets, col_mask)
                    total_loss = (
                        float(args.block_margin_weight) * structured_loss
                        + float(args.edge_bce_weight) * edge_loss
                        + float(args.row_ce_weight) * row_loss
                        + float(args.col_bce_weight) * col_loss
                    )
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                val_metrics = evaluate(
                    model=model,
                    loader=val_loader,
                    device=device,
                    feature_indices=feature_indices,
                    margin=float(args.block_margin),
                )
                metric_row = {"epoch": int(epoch), **val_metrics}
                handle.write(json.dumps(metric_row))
                handle.write("\n")
                handle.flush()

                metric = float(val_metrics["ambiguous_exact_block_acc"]) + 0.25 * float(val_metrics["ambiguous_row_top1"])
                if metric >= best_metric:
                    best_metric = metric
                    torch.save(
                        {
                            "family": MODEL_FAMILY,
                            "model_state": model.state_dict(),
                            "feature_mode": str(args.feature_mode),
                            "model_variant": str(args.model_variant),
                            "init_checkpoint": str(args.init_checkpoint or ""),
                            "init_loaded_tensor_count": int(summary_row.get("init_loaded_tensor_count", 0)),
                            "edge_logit_mode": str(args.edge_logit_mode),
                            "use_base_residual": bool(use_base_residual),
                            "base_score_index": int(base_score_index),
                            "base_logit_scale": float(args.base_logit_scale),
                            "side_init_scale": float(args.side_init_scale),
                            "feature_names": list(kept_feature_names),
                            "edge_aux_feature_names": list(edge_aux_feature_names),
                            "all_edge_feature_names": list(all_feature_names),
                            "row_feature_names": list(first_block.row_feature_names),
                            "row_aux_feature_names": list(row_aux_feature_names),
                            "col_feature_names": list(first_block.col_feature_names),
                            "col_aux_feature_names": list(col_aux_feature_names),
                            "input_dim": int(input_dim),
                            "row_context_dim": int(len(first_block.row_feature_names)),
                            "col_context_dim": int(len(first_block.col_feature_names)),
                            "hidden_dim": int(args.hidden_dim),
                            "stage_embed_dim": int(args.stage_embed_dim),
                            "num_heads": int(args.num_heads),
                            "num_layers": int(args.num_layers),
                            "dropout": float(args.dropout),
                            "num_stages": int(default_num_stages()),
                            "block_margin": float(args.block_margin),
                        },
                        best_ckpt,
                    )
                    summary_row.update(
                        {
                            "best_epoch": int(epoch),
                            "best_metric": float(metric),
                            "val_exact_block_acc": float(val_metrics["exact_block_acc"]),
                            "val_ambiguous_exact_block_acc": float(val_metrics["ambiguous_exact_block_acc"]),
                            "val_row_top1": float(val_metrics["row_top1"]),
                            "val_ambiguous_row_top1": float(val_metrics["ambiguous_row_top1"]),
                            "val_structured_loss": float(val_metrics["structured_loss"]),
                            "val_edge_bce": float(val_metrics["edge_bce"]),
                            "val_row_ce": float(val_metrics["row_ce"]),
                            "val_col_bce": float(val_metrics["col_bce"]),
                        }
                    )
                    write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)

        summary_row["status"] = "success"
        write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
        append_registry(args, summary_csv, best_ckpt, "success", "FGAS true block matcher training")
        return 0
    except Exception as exc:
        summary_row["status"] = "failed"
        summary_row["error"] = repr(exc)
        write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
        append_registry(args, summary_csv, best_ckpt, "failed", "FGAS true block matcher training")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
