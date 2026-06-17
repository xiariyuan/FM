#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
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
from projects.fgas.fgas.model.block_resolver import FGASBlockResolver, STAGE_NAME_TO_ID
from projects.fgas.fgas.model.block_resolver_v2 import FGASAssociationResolverV2
from projects.fgas.fgas.model.block_resolver_v3_trackquery import FGASAssociationResolverV3TrackQuery


REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"
SUMMARY_FIELDS = [
    "train_jsonl",
    "val_jsonl",
    "arch",
    "feature_mode",
    "input_dim",
    "epochs",
    "batch_size",
    "lr",
    "weight_decay",
    "hidden_dim",
    "stage_embed_dim",
    "num_heads",
    "num_attn_layers",
    "ambiguous_oversample",
    "col_bce_weight",
    "row_context_dim",
    "col_context_dim",
    "best_epoch",
    "best_metric",
    "val_row_top1",
    "val_ambiguous_row_top1",
    "val_edge_bce",
    "val_row_ce",
    "val_col_bce",
    "status",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the FGAS block resolver on block-bank JSONL files.")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--arch", choices=["v1", "v2_trackdet", "v3_trackquery"], default="v1")
    parser.add_argument("--feature-mode", choices=["full", "nofreq"], default="full")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--stage-embed-dim", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-attn-layers", type=int, default=2)
    parser.add_argument("--edge-bce-weight", type=float, default=1.0)
    parser.add_argument("--row-ce-weight", type=float, default=1.0)
    parser.add_argument("--col-bce-weight", type=float, default=0.25)
    parser.add_argument("--no-match-weight", type=float, default=0.5)
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
        "scripts/train_fgas_block_resolver.py",
        "--dataset",
        "MOT17",
        "--split",
        "blockbank",
        "--tracker-family",
        "botsort_fgas",
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
    subprocess.run(cmd, check=False)


def set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def stage_ids_from_names(stage_names: List[str], device: torch.device) -> torch.Tensor:
    ids = [int(STAGE_NAME_TO_ID.get(str(name), STAGE_NAME_TO_ID["primary"])) for name in stage_names]
    return torch.tensor(ids, dtype=torch.long, device=device)


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
    if not flat_mask.any():
        return flat_logits.sum() * 0.0
    return nn.functional.cross_entropy(flat_logits[flat_mask], flat_targets[flat_mask])


def edge_bce_loss(edge_logits: torch.Tensor, edge_labels: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
    if not edge_mask.any():
        return edge_logits.sum() * 0.0
    return nn.functional.binary_cross_entropy_with_logits(edge_logits[edge_mask], edge_labels[edge_mask])


def col_bce_loss(
    col_logits: torch.Tensor | None,
    col_targets: torch.Tensor,
    col_mask: torch.Tensor,
    fallback: torch.Tensor,
) -> torch.Tensor:
    if col_logits is None or not col_mask.any():
        return fallback.sum() * 0.0
    return nn.functional.binary_cross_entropy_with_logits(col_logits[col_mask], col_targets[col_mask])


def evaluate(
    model: FGASBlockResolver,
    loader: DataLoader,
    device: torch.device,
    feature_indices: torch.Tensor,
) -> Dict[str, float]:
    model.eval()
    total_rows = 0
    correct_rows = 0
    ambiguous_total_rows = 0
    ambiguous_correct_rows = 0
    total_edge_bce = 0.0
    total_row_ce = 0.0
    total_col_bce = 0.0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            edge_features = batch["edge_features"].to(device)
            edge_features = edge_features.index_select(dim=-1, index=feature_indices)
            edge_labels = batch["edge_labels"].to(device)
            edge_mask = batch["edge_mask"].to(device)
            row_mask = batch["row_mask"].to(device)
            row_targets = batch["row_targets"].to(device)
            col_mask = batch["col_mask"].to(device)
            col_targets = batch["col_newborn_targets"].to(device)
            ambiguous = batch["ambiguous"].to(device)
            stage_ids = stage_ids_from_names(list(batch["stage_names"]), device)
            forward_kwargs = {
                "edge_features": edge_features,
                "edge_mask": edge_mask,
                "stage_ids": stage_ids,
            }
            if batch["row_features"].shape[-1] > 0:
                forward_kwargs["row_context"] = batch["row_features"].to(device)
            if batch["col_features"].shape[-1] > 0:
                forward_kwargs["col_context"] = batch["col_features"].to(device)
            output = model(**forward_kwargs)
            edge_loss = edge_bce_loss(output.edge_logits, edge_labels, edge_mask)
            row_loss = row_ce_loss(output.edge_logits, output.row_no_match_logits, row_targets, row_mask)
            col_loss = col_bce_loss(output.col_newborn_logits, col_targets, col_mask, output.edge_logits)
            total_edge_bce += float(edge_loss.item())
            total_row_ce += float(row_loss.item())
            total_col_bce += float(col_loss.item())
            batches += 1

            combined_logits = torch.cat([output.edge_logits, output.row_no_match_logits.unsqueeze(-1)], dim=-1)
            predictions = combined_logits.argmax(dim=-1)
            no_match_index = output.edge_logits.shape[-1]
            target = row_targets.clone()
            target[target < 0] = no_match_index
            valid_rows = row_mask
            correct = (predictions == target) & valid_rows
            correct_rows += int(correct.sum().item())
            total_rows += int(valid_rows.sum().item())
            if ambiguous.any():
                amb_mask = valid_rows & ambiguous.unsqueeze(-1)
                ambiguous_correct_rows += int((correct & amb_mask).sum().item())
                ambiguous_total_rows += int(amb_mask.sum().item())

    return {
        "row_top1": float(correct_rows / total_rows) if total_rows else 0.0,
        "ambiguous_row_top1": float(ambiguous_correct_rows / ambiguous_total_rows) if ambiguous_total_rows else 0.0,
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
        "arch": str(args.arch),
        "feature_mode": str(args.feature_mode),
        "input_dim": 0,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "hidden_dim": int(args.hidden_dim),
        "stage_embed_dim": int(args.stage_embed_dim),
        "num_heads": int(args.num_heads),
        "num_attn_layers": int(args.num_attn_layers),
        "ambiguous_oversample": float(args.ambiguous_oversample),
        "col_bce_weight": float(args.col_bce_weight),
        "row_context_dim": 0,
        "col_context_dim": 0,
        "best_epoch": -1,
        "best_metric": 0.0,
        "val_row_top1": 0.0,
        "val_ambiguous_row_top1": 0.0,
        "val_edge_bce": math.inf,
        "val_row_ce": math.inf,
        "val_col_bce": math.inf,
        "status": "running",
        "error": "",
    }
    write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
    append_registry(args, summary_csv, best_ckpt, "running", "FGAS block resolver training")

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
        row_feature_names = list(first_block.row_feature_names)
        col_feature_names = list(first_block.col_feature_names)
        if str(args.feature_mode) == "nofreq":
            kept_feature_names = [name for name in all_feature_names if name not in {"s_low", "s_mid", "s_high"}]
        else:
            kept_feature_names = list(all_feature_names)
        feature_indices_list = [all_feature_names.index(name) for name in kept_feature_names]
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
        summary_row["input_dim"] = int(input_dim)
        summary_row["row_context_dim"] = int(len(row_feature_names))
        summary_row["col_context_dim"] = int(len(col_feature_names))
        if str(args.arch) == "v3_trackquery":
            model = FGASAssociationResolverV3TrackQuery(
                input_dim=input_dim,
                row_context_dim=len(row_feature_names),
                col_context_dim=len(col_feature_names),
                hidden_dim=int(args.hidden_dim),
                stage_embed_dim=int(args.stage_embed_dim),
                num_stages=len(STAGE_NAME_TO_ID),
                num_heads=int(args.num_heads),
                num_layers=int(args.num_attn_layers),
            ).to(device)
        elif str(args.arch) == "v2_trackdet":
            model = FGASAssociationResolverV2(
                input_dim=input_dim,
                hidden_dim=int(args.hidden_dim),
                stage_embed_dim=int(args.stage_embed_dim),
                num_stages=len(STAGE_NAME_TO_ID),
                num_heads=int(args.num_heads),
                num_layers=int(args.num_attn_layers),
            ).to(device)
        else:
            model = FGASBlockResolver(
                input_dim=input_dim,
                hidden_dim=int(args.hidden_dim),
                stage_embed_dim=int(args.stage_embed_dim),
                num_stages=len(STAGE_NAME_TO_ID),
            ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )

        best_metric = -1.0
        metrics_handle = metrics_jsonl.open("w", encoding="utf-8")
        try:
            for epoch in range(1, int(args.epochs) + 1):
                model.train()
                for batch in train_loader:
                    edge_features = batch["edge_features"].to(device)
                    edge_features = edge_features.index_select(dim=-1, index=feature_indices)
                    edge_labels = batch["edge_labels"].to(device)
                    edge_mask = batch["edge_mask"].to(device)
                    row_mask = batch["row_mask"].to(device)
                    row_targets = batch["row_targets"].to(device)
                    col_mask = batch["col_mask"].to(device)
                    col_targets = batch["col_newborn_targets"].to(device)
                    stage_ids = stage_ids_from_names(list(batch["stage_names"]), device)
                    forward_kwargs = {
                        "edge_features": edge_features,
                        "edge_mask": edge_mask,
                        "stage_ids": stage_ids,
                    }
                    if batch["row_features"].shape[-1] > 0:
                        forward_kwargs["row_context"] = batch["row_features"].to(device)
                    if batch["col_features"].shape[-1] > 0:
                        forward_kwargs["col_context"] = batch["col_features"].to(device)
                    output = model(**forward_kwargs)
                    edge_loss = edge_bce_loss(output.edge_logits, edge_labels, edge_mask)
                    row_loss = row_ce_loss(output.edge_logits, output.row_no_match_logits, row_targets, row_mask)
                    col_loss = col_bce_loss(output.col_newborn_logits, col_targets, col_mask, output.edge_logits)
                    loss = (
                        float(args.edge_bce_weight) * edge_loss
                        + float(args.row_ce_weight) * row_loss
                        + float(args.col_bce_weight) * col_loss
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                val_metrics = evaluate(model, val_loader, device, feature_indices)
                record = {"epoch": epoch, **val_metrics}
                metrics_handle.write(json.dumps(record))
                metrics_handle.write("\n")
                metrics_handle.flush()

                metric = float(val_metrics["ambiguous_row_top1"])
                if metric >= best_metric:
                    best_metric = metric
                    torch.save(
                        {
                            "model_state": model.state_dict(),
                            "arch": str(args.arch),
                            "input_dim": int(input_dim),
                            "hidden_dim": int(args.hidden_dim),
                            "stage_embed_dim": int(args.stage_embed_dim),
                            "num_heads": int(args.num_heads),
                            "num_attn_layers": int(args.num_attn_layers),
                            "stage_name_to_id": dict(STAGE_NAME_TO_ID),
                            "feature_mode": str(args.feature_mode),
                            "feature_names": list(kept_feature_names),
                            "row_context_dim": int(len(row_feature_names)),
                            "col_context_dim": int(len(col_feature_names)),
                            "row_feature_names": list(row_feature_names),
                            "col_feature_names": list(col_feature_names),
                        },
                        best_ckpt,
                    )
                    summary_row.update(
                        {
                            "best_epoch": int(epoch),
                            "best_metric": float(metric),
                            "val_row_top1": float(val_metrics["row_top1"]),
                            "val_ambiguous_row_top1": float(val_metrics["ambiguous_row_top1"]),
                            "val_edge_bce": float(val_metrics["edge_bce"]),
                            "val_row_ce": float(val_metrics["row_ce"]),
                            "val_col_bce": float(val_metrics["col_bce"]),
                        }
                    )
                    write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
        finally:
            metrics_handle.close()

        summary_row["status"] = "success"
        write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
        append_registry(args, summary_csv, best_ckpt, "success", "FGAS block resolver training")
        return 0
    except Exception as exc:
        summary_row["status"] = "failed"
        summary_row["error"] = repr(exc)
        write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
        append_registry(args, summary_csv, best_ckpt, "failed", "FGAS block resolver training")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
