#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.fgas.fgas.data.block_types import ConflictBlock
from projects.fgas.fgas.data.blockbank_io import load_blockbank_jsonl
from projects.fgas.fgas.features.block_gate_features import BLOCK_GATE_FEATURE_NAMES, build_block_gate_feature_vector
from projects.fgas.fgas.features.edge_features import DOMAIN_FEATURE_NAMES, build_domain_feature_vector
from projects.fgas.fgas.model.pair_scorer import FGASPairScorer


REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"
SUMMARY_FIELDS = [
    "blockbank_jsonl",
    "pair_scorer_checkpoint",
    "mode",
    "feature_dim",
    "blocks_total",
    "rows_total",
    "changed_blocks",
    "written_blocks",
    "positive_blocks",
    "negative_blocks",
    "neutral_blocks",
    "beneficial_blocks",
    "harmful_blocks",
    "status",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build block-level gate examples from a blockbank and trained FGAS pair scorer.")
    parser.add_argument("--blockbank-jsonl", required=True)
    parser.add_argument("--pair-scorer-checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--keep-neutral", action="store_true", help="keep neutral blocks in the output dataset")
    parser.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    return parser.parse_args()


def write_single_row_csv(path: Path, fieldnames: Sequence[str], row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
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
        "scripts/build_fgas_block_gate_dataset.py",
        "--dataset",
        "MOT17",
        "--split",
        "blockbank_block_gate",
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
        str(Path(args.pair_scorer_checkpoint)),
        "--notes",
        notes,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def _edge_feature_map(block: ConflictBlock, edge_idx: int) -> Dict[str, float]:
    edge = block.edges[edge_idx]
    return {
        str(name): float(value)
        for name, value in zip(edge.feature_names, edge.features)
    }


def _row_feature_map(block: ConflictBlock, row_idx: int) -> Dict[str, float]:
    if row_idx < 0 or row_idx >= len(block.row_features):
        return {}
    return {
        str(name): float(value)
        for name, value in zip(block.row_feature_names, block.row_features[row_idx])
    }


def _col_feature_map(block: ConflictBlock, col_idx: int) -> Dict[str, float]:
    if col_idx < 0 or col_idx >= len(block.col_features):
        return {}
    return {
        str(name): float(value)
        for name, value in zip(block.col_feature_names, block.col_features[col_idx])
    }


def row_top1_and_margin(matrix: np.ndarray, valid_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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


def load_pair_scorer(checkpoint_path: Path, device: torch.device) -> Tuple[FGASPairScorer, Dict[str, object]]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    model = FGASPairScorer(
        input_dim=int(payload.get("input_dim", 0)),
        hidden_dim=int(payload.get("hidden_dim", 64)),
        dropout=float(payload.get("dropout", 0.0)),
    )
    model.load_state_dict(payload["model_state"])
    model.eval()
    model.to(device)
    return model, payload


def build_pair_prob_matrix(
    *,
    block: ConflictBlock,
    model: FGASPairScorer,
    device: torch.device,
    edge_feature_names: Sequence[str],
    row_feature_names: Sequence[str],
    col_feature_names: Sequence[str],
    domain_feature_names: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_count = len(block.row_track_ids)
    col_count = len(block.col_det_ids)
    base_similarity = np.zeros((row_count, col_count), dtype=np.float32)
    probs = np.zeros((row_count, col_count), dtype=np.float32)
    valid_mask = np.zeros((row_count, col_count), dtype=bool)
    seq_name = str(block.metadata.get("seq_name", ""))
    domain_map = {
        name: float(value)
        for name, value in zip(DOMAIN_FEATURE_NAMES, build_domain_feature_vector(seq_name))
    }
    vectors: List[List[float]] = []
    positions: List[Tuple[int, int]] = []

    for edge_idx, edge in enumerate(block.edges):
        if not bool(edge.valid):
            continue
        row_idx = int(edge.row_index)
        col_idx = int(edge.col_index)
        edge_map = _edge_feature_map(block, edge_idx)
        row_map = _row_feature_map(block, row_idx)
        col_map = _col_feature_map(block, col_idx)
        values = [float(edge_map.get(name, 0.0)) for name in edge_feature_names]
        values.extend(float(row_map.get(name, 0.0)) for name in row_feature_names)
        values.extend(float(col_map.get(name, 0.0)) for name in col_feature_names)
        values.extend(float(domain_map.get(name, 0.0)) for name in domain_feature_names)
        vectors.append(values)
        positions.append((row_idx, col_idx))
        valid_mask[row_idx, col_idx] = True
        base_similarity[row_idx, col_idx] = float(edge_map.get("base_similarity", 0.0))

    if vectors:
        with torch.no_grad():
            inputs = torch.tensor(np.asarray(vectors, dtype=np.float32), dtype=torch.float32, device=device)
            logits = model(inputs)
            pair_probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
        for idx, (row_idx, col_idx) in enumerate(positions):
            probs[row_idx, col_idx] = float(pair_probs[idx])

    return probs, base_similarity, valid_mask


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary.csv"
    dataset_jsonl = out_dir / "block_gate_dataset.jsonl"
    summary_row: Dict[str, object] = {
        "blockbank_jsonl": str(Path(args.blockbank_jsonl)),
        "pair_scorer_checkpoint": str(Path(args.pair_scorer_checkpoint)),
        "mode": "",
        "feature_dim": int(len(BLOCK_GATE_FEATURE_NAMES)),
        "blocks_total": 0,
        "rows_total": 0,
        "changed_blocks": 0,
        "written_blocks": 0,
        "positive_blocks": 0,
        "negative_blocks": 0,
        "neutral_blocks": 0,
        "beneficial_blocks": 0,
        "harmful_blocks": 0,
        "status": "running",
        "error": "",
    }
    write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
    append_registry(args, summary_csv, "running", "building FGAS block gate dataset")

    try:
        device = torch.device(str(args.device))
        model, payload = load_pair_scorer(Path(args.pair_scorer_checkpoint), device)
        summary_row["mode"] = str(payload.get("mode", "nofreq"))
        blocks = load_blockbank_jsonl(Path(args.blockbank_jsonl))
        summary_row["blocks_total"] = int(len(blocks))
        write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)

        edge_feature_names = list(payload.get("edge_feature_names", []))
        row_feature_names = list(payload.get("row_feature_names", []))
        col_feature_names = list(payload.get("col_feature_names", []))
        domain_feature_names = list(payload.get("domain_feature_names", DOMAIN_FEATURE_NAMES))

        changed_blocks = 0
        written_blocks = 0
        positive_blocks = 0
        negative_blocks = 0
        neutral_blocks = 0
        beneficial_blocks = 0
        harmful_blocks = 0
        rows_total = 0

        with dataset_jsonl.open("w", encoding="utf-8") as handle:
            for block in blocks:
                if not block.edges:
                    continue
                probs, base_similarity, valid_mask = build_pair_prob_matrix(
                    block=block,
                    model=model,
                    device=device,
                    edge_feature_names=edge_feature_names,
                    row_feature_names=row_feature_names,
                    col_feature_names=col_feature_names,
                    domain_feature_names=domain_feature_names,
                )
                row_count = len(block.row_track_ids)
                rows_total += int(row_count)
                if row_count == 0 or not np.any(valid_mask):
                    continue

                base_best, base_row_margin = row_top1_and_margin(base_similarity, valid_mask)
                refined_best, refined_row_margin = row_top1_and_margin(probs, valid_mask)
                changed_mask = (base_best >= 0) & (refined_best >= 0) & (base_best != refined_best)
                if np.any(changed_mask):
                    changed_blocks += 1

                base_correct_rows = 0
                refined_correct_rows = 0
                beneficial_rows = 0
                harmful_rows = 0
                for row_idx, target_col in enumerate(block.row_match_targets):
                    target_col = int(target_col)
                    base_correct = int(target_col >= 0 and int(base_best[row_idx]) == target_col)
                    refined_correct = int(target_col >= 0 and int(refined_best[row_idx]) == target_col)
                    base_correct_rows += base_correct
                    refined_correct_rows += refined_correct
                    beneficial_rows += int(base_correct == 0 and refined_correct == 1)
                    harmful_rows += int(base_correct == 1 and refined_correct == 0)

                if beneficial_rows > 0:
                    beneficial_blocks += 1
                if harmful_rows > 0:
                    harmful_blocks += 1

                delta_correct_rows = int(refined_correct_rows - base_correct_rows)
                if delta_correct_rows > 0:
                    label = 1
                    flip_type = "beneficial"
                    positive_blocks += 1
                elif delta_correct_rows < 0:
                    label = 0
                    flip_type = "harmful"
                    negative_blocks += 1
                else:
                    neutral_blocks += 1
                    if not bool(args.keep_neutral):
                        continue
                    label = 0
                    flip_type = "neutral"
                    negative_blocks += 1

                block_ambiguous_flag = float(
                    bool(block.ambiguous)
                    or np.any((base_best >= 0) & (base_row_margin < 0.05))
                )
                row_features = (
                    np.asarray(block.row_features, dtype=np.float32)
                    if block.row_features
                    else np.zeros((row_count, 0), dtype=np.float32)
                )
                features = build_block_gate_feature_vector(
                    row_feature_names=block.row_feature_names,
                    row_features=row_features,
                    valid_mask=valid_mask,
                    base_similarity=base_similarity,
                    probs=probs,
                    base_best=base_best,
                    refined_best=refined_best,
                    base_row_margin=base_row_margin,
                    refined_row_margin=refined_row_margin,
                    seq_name=str(block.metadata.get("seq_name", "")),
                    block_ambiguous_flag=block_ambiguous_flag,
                )
                record = {
                    "block_key": str(block.block_key),
                    "seq_name": str(block.metadata.get("seq_name", "")),
                    "frame_id": int(block.metadata.get("frame_id", -1)),
                    "label": int(label),
                    "flip_type": flip_type,
                    "base_correct_rows": int(base_correct_rows),
                    "refined_correct_rows": int(refined_correct_rows),
                    "delta_correct_rows": int(delta_correct_rows),
                    "beneficial_rows": int(beneficial_rows),
                    "harmful_rows": int(harmful_rows),
                    "changed_rows": int(np.count_nonzero(changed_mask)),
                    "feature_names": list(BLOCK_GATE_FEATURE_NAMES),
                    "features": [float(value) for value in features],
                }
                handle.write(json.dumps(record))
                handle.write("\n")
                written_blocks += 1

        summary_row.update(
            {
                "rows_total": int(rows_total),
                "changed_blocks": int(changed_blocks),
                "written_blocks": int(written_blocks),
                "positive_blocks": int(positive_blocks),
                "negative_blocks": int(negative_blocks),
                "neutral_blocks": int(neutral_blocks),
                "beneficial_blocks": int(beneficial_blocks),
                "harmful_blocks": int(harmful_blocks),
                "status": "success",
            }
        )
        write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
        append_registry(args, summary_csv, "success", "built FGAS block gate dataset")
        return 0
    except Exception as exc:
        summary_row["status"] = "failed"
        summary_row["error"] = repr(exc)
        write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
        append_registry(args, summary_csv, "failed", "building FGAS block gate dataset")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
