#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.fgas.fgas.data.block_types import collate_conflict_blocks
from projects.fgas.fgas.data.blockbank_io import load_blockbank_jsonl
from projects.fgas.fgas.features.acceptance_features import ACCEPTANCE_FEATURE_NAMES, build_acceptance_feature_vector
from projects.fgas.fgas.model.block_primitive import FGASAmbiguityBlockPrimitive
from projects.fgas.fgas.model.block_resolver import FGASBlockResolver, STAGE_NAME_TO_ID
from projects.fgas.fgas.model.block_resolver_v2 import FGASAssociationResolverV2
from projects.fgas.fgas.model.block_resolver_v3_trackquery import FGASAssociationResolverV3TrackQuery


REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"
SUMMARY_FIELDS = [
    "blockbank_jsonl",
    "checkpoint",
    "arch",
    "feature_mode",
    "rows_total",
    "blocks_total",
    "changed_rows",
    "positive_rows",
    "negative_rows",
    "harmful_rows",
    "neutral_rows",
    "status",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build changed-row acceptance examples from a blockbank and trained FGAS resolver.")
    parser.add_argument("--blockbank-jsonl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    return parser.parse_args()


def write_single_row_csv(path: Path, fieldnames: List[str], row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def append_registry(args: argparse.Namespace, summary_csv: Path, status: str, notes: str) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "append_experiment_record.py"),
        "--csv",
        str(args.registry_csv),
        "--kind",
        "analysis",
        "--status",
        status,
        "--script",
        "scripts/build_fgas_acceptance_dataset.py",
        "--dataset",
        "MOT17",
        "--split",
        "blockbank_acceptance",
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
        str(Path(args.checkpoint)),
        "--notes",
        notes,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def stage_ids_from_names(stage_names: List[str], device: torch.device) -> torch.Tensor:
    ids = [int(STAGE_NAME_TO_ID.get(str(name), STAGE_NAME_TO_ID["primary"])) for name in stage_names]
    return torch.tensor(ids, dtype=torch.long, device=device)


def row_top1_and_margin(matrix: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    row_best = np.full((matrix.shape[0],), fill_value=-1, dtype=int)
    row_margin = np.zeros((matrix.shape[0],), dtype=np.float32)
    for row_idx in range(matrix.shape[0]):
        valid_cols = np.where(valid_mask[row_idx])[0]
        if valid_cols.size == 0:
            continue
        row_scores = np.asarray(matrix[row_idx, valid_cols], dtype=np.float32)
        order = np.argsort(row_scores)[::-1]
        best_local = int(valid_cols[order[0]])
        row_best[row_idx] = best_local
        best_score = float(row_scores[order[0]])
        second_score = float(row_scores[order[1]]) if order.size > 1 else 0.0
        row_margin[row_idx] = float(best_score - second_score)
    return row_best, row_margin


def load_model(checkpoint_path: Path, device: torch.device):
    payload = torch.load(checkpoint_path, map_location="cpu")
    family = str(payload.get("family", ""))
    if family == "fgas_block_primitive":
        model = FGASAmbiguityBlockPrimitive(
            input_dim=int(payload.get("input_dim", 0)),
            row_context_dim=int(payload.get("row_context_dim", 0)),
            col_context_dim=int(payload.get("col_context_dim", 0)),
            hidden_dim=int(payload.get("hidden_dim", 128)),
            stage_embed_dim=int(payload.get("stage_embed_dim", 16)),
            num_stages=int(payload.get("num_stages", len(STAGE_NAME_TO_ID))),
            num_heads=int(payload.get("num_heads", 4)),
            num_layers=int(payload.get("num_layers", 2)),
            dropout=float(payload.get("dropout", 0.0)),
        )
        model.load_state_dict(payload["model_state"])
        model.to(device)
        model.eval()
        return model, payload
    arch = str(payload.get("arch", "v1"))
    input_dim = int(payload.get("input_dim", 0))
    hidden_dim = int(payload.get("hidden_dim", 64))
    stage_embed_dim = int(payload.get("stage_embed_dim", 8))
    if arch == "v3_trackquery":
        model = FGASAssociationResolverV3TrackQuery(
            input_dim=input_dim,
            row_context_dim=int(payload.get("row_context_dim", 0)),
            col_context_dim=int(payload.get("col_context_dim", 0)),
            hidden_dim=hidden_dim,
            stage_embed_dim=stage_embed_dim,
            num_stages=len(STAGE_NAME_TO_ID),
            num_heads=int(payload.get("num_heads", 4)),
            num_layers=int(payload.get("num_attn_layers", 2)),
        )
    elif arch == "v2_trackdet":
        model = FGASAssociationResolverV2(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            stage_embed_dim=stage_embed_dim,
            num_stages=len(STAGE_NAME_TO_ID),
            num_heads=int(payload.get("num_heads", 4)),
            num_layers=int(payload.get("num_attn_layers", 2)),
        )
    else:
        model = FGASBlockResolver(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            stage_embed_dim=stage_embed_dim,
            num_stages=len(STAGE_NAME_TO_ID),
        )
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model, payload


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary.csv"
    dataset_jsonl = out_dir / "acceptance_dataset.jsonl"
    summary_row: Dict[str, object] = {
        "blockbank_jsonl": str(Path(args.blockbank_jsonl)),
        "checkpoint": str(Path(args.checkpoint)),
        "arch": "",
        "feature_mode": "",
        "rows_total": 0,
        "blocks_total": 0,
        "changed_rows": 0,
        "positive_rows": 0,
        "negative_rows": 0,
        "harmful_rows": 0,
        "neutral_rows": 0,
        "status": "running",
        "error": "",
    }
    write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
    append_registry(args, summary_csv, "running", "building FGAS acceptance dataset")

    try:
        device = torch.device(str(args.device))
        model, payload = load_model(Path(args.checkpoint), device)
        feature_names = list(payload.get("feature_names", []))
        row_feature_names = list(payload.get("row_feature_names", []))
        col_feature_names = list(payload.get("col_feature_names", []))
        summary_row["arch"] = str(payload.get("family", payload.get("arch", "v1")))
        summary_row["feature_mode"] = str(payload.get("feature_mode", "full"))
        blocks = load_blockbank_jsonl(Path(args.blockbank_jsonl))
        summary_row["blocks_total"] = int(len(blocks))
        write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)

        positive_rows = 0
        negative_rows = 0
        harmful_rows = 0
        neutral_rows = 0
        changed_rows = 0
        rows_total = 0

        with dataset_jsonl.open("w", encoding="utf-8") as handle:
            for block in blocks:
                batch = collate_conflict_blocks([block])
                full_edge_names = list(block.edges[0].feature_names) if block.edges else []
                if not full_edge_names:
                    continue
                edge_features_full = batch["edge_features"][0].cpu().numpy()
                edge_mask = batch["edge_mask"][0].cpu().numpy().astype(bool)
                row_features_full = batch["row_features"][0].cpu().numpy()
                col_features_full = batch["col_features"][0].cpu().numpy()
                row_targets = batch["row_targets"][0].cpu().numpy()
                rows_total += int(batch["row_mask"][0].sum().item())

                feature_indices = torch.tensor(
                    [full_edge_names.index(name) for name in feature_names],
                    dtype=torch.long,
                    device=device,
                )
                edge_features = batch["edge_features"].to(device).index_select(dim=-1, index=feature_indices)
                forward_kwargs = {
                    "edge_features": edge_features,
                    "edge_mask": batch["edge_mask"].to(device),
                    "stage_ids": stage_ids_from_names(list(batch["stage_names"]), device),
                }
                if str(payload.get("family", "")) == "fgas_block_primitive" or row_features_full.shape[-1] > 0:
                    forward_kwargs["row_context"] = batch["row_features"].to(device)
                if str(payload.get("family", "")) == "fgas_block_primitive" or col_features_full.shape[-1] > 0:
                    forward_kwargs["col_context"] = batch["col_features"].to(device)
                with torch.no_grad():
                    output = model(**forward_kwargs)
                probs = torch.sigmoid(output.edge_logits)[0].cpu().numpy()
                row_nomatch_probs = torch.sigmoid(output.row_no_match_logits)[0].cpu().numpy()

                base_idx = full_edge_names.index("base_similarity")
                base_similarity = edge_features_full[:, :, base_idx]
                base_best, base_row_margin = row_top1_and_margin(base_similarity, edge_mask)
                refined_best, refined_row_margin = row_top1_and_margin(probs, edge_mask)

                for row_idx in range(len(block.row_track_ids)):
                    base_col = int(base_best[row_idx])
                    refined_col = int(refined_best[row_idx])
                    if base_col < 0 or refined_col < 0 or base_col == refined_col:
                        continue
                    changed_rows += 1
                    target_col = int(row_targets[row_idx])
                    base_correct = int(target_col >= 0 and base_col == target_col)
                    refined_correct = int(target_col >= 0 and refined_col == target_col)
                    label = int(refined_correct == 1 and base_correct == 0)
                    if label == 1:
                        positive_rows += 1
                        flip_type = "beneficial"
                    else:
                        negative_rows += 1
                        if base_correct == 1 and refined_correct == 0:
                            harmful_rows += 1
                            flip_type = "harmful"
                        else:
                            neutral_rows += 1
                            flip_type = "neutral"
                    features = build_acceptance_feature_vector(
                        edge_feature_names=full_edge_names,
                        row_feature_names=row_feature_names,
                        col_feature_names=col_feature_names,
                        edge_features=edge_features_full,
                        row_features=row_features_full,
                        col_features=col_features_full,
                        valid_mask=edge_mask,
                        probs=probs,
                        row_nomatch_probs=row_nomatch_probs,
                        base_best=base_best,
                        refined_best=refined_best,
                        base_row_margin=base_row_margin,
                        refined_row_margin=refined_row_margin,
                        row_idx=row_idx,
                    )
                    record = {
                        "block_key": str(block.block_key),
                        "seq_name": str(block.metadata.get("seq_name", "")),
                        "frame_id": int(block.metadata.get("frame_id", -1)),
                        "row_index": int(row_idx),
                        "track_gt_id": int(block.row_track_ids[row_idx]),
                        "target_col": int(target_col),
                        "base_best": int(base_col),
                        "refined_best": int(refined_col),
                        "base_correct": int(base_correct),
                        "refined_correct": int(refined_correct),
                        "flip_type": flip_type,
                        "label": int(label),
                        "feature_names": list(ACCEPTANCE_FEATURE_NAMES),
                        "features": [float(v) for v in features],
                    }
                    handle.write(json.dumps(record))
                    handle.write("\n")

        summary_row.update(
            {
                "rows_total": int(rows_total),
                "changed_rows": int(changed_rows),
                "positive_rows": int(positive_rows),
                "negative_rows": int(negative_rows),
                "harmful_rows": int(harmful_rows),
                "neutral_rows": int(neutral_rows),
                "status": "success",
            }
        )
        write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
        append_registry(args, summary_csv, "success", "built FGAS acceptance dataset")
        return 0
    except Exception as exc:
        summary_row["status"] = "failed"
        summary_row["error"] = repr(exc)
        write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
        append_registry(args, summary_csv, "failed", "building FGAS acceptance dataset")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
